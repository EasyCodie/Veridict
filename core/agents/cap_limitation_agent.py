"""Cap & Limitation Agent - Priority 2 Worker in Micro-Agent Factory.

This agent extracts financial caps, liability limitations, and exceptions from
liability-related contract clauses. It uses GPT-5 Nano with deterministic
settings for consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
- Cap amount parsing (numeric and relative)
- Comprehensive cost tracking
- Retry logic with exponential backoff
- PostgreSQL integration for fetch/store
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, BeforeValidator, field_validator
from typing_extensions import Annotated

from app.config import get_settings
from core.cost_tracker import CostTracker, ExecutionStatus
from core.agents.utils.cap_parser import parse_cap_amount, detect_relative_cap
from core.agents.prompts.cap_limitation_prompt import (
    CAP_LIMITATION_SYSTEM_PROMPT,
    CAP_LIMITATION_USER_PROMPT_TEMPLATE,
)


def clamp_confidence(v: float) -> float:
    """Clamp confidence value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


# Type alias for clamped confidence values
ClampedConfidence = Annotated[float, BeforeValidator(clamp_confidence)]


logger = logging.getLogger("veridict.cap_limitation_agent")


class CapType(str, Enum):
    """Types of caps that can be extracted."""
    LIABILITY_CAP = "liability_cap"
    INDEMNITY_CAP = "indemnity_cap"
    DAMAGES_LIMITATION = "damages_limitation"
    DEDUCTIBLE = "deductible"


class RetryableError(str, Enum):
    """Error types that should trigger a retry."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"


class NonRetryableError(str, Enum):
    """Error types that should NOT trigger a retry."""
    VALIDATION_ERROR = "validation_error"
    MALFORMED_REQUEST = "malformed_request"


class ClauseMetadata(BaseModel):
    """Metadata about the contract containing the clause."""
    contract_type: str = Field(default="", description="Type of contract (e.g., 'SPA', 'NDA')")
    parties: list[str] = Field(default_factory=list, description="Parties to the contract")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    execution_date: str | None = Field(default=None, description="Execution date (ISO8601)")


class CapLimitationInput(BaseModel):
    """Input contract for the Cap & Limitation Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    clause_type: str = Field(default="", description="Type of clause (optional)")
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class Cap(BaseModel):
    """A single extracted cap or limitation."""
    cap_type: str = Field(..., description="Type of cap (liability_cap, indemnity_cap, etc.)")
    cap_amount: str = Field(..., description="Cap amount as string (e.g., '$5M' or '15% of Purchase Price')")
    cap_amount_numeric: float | None = Field(default=None, description="Numeric amount in USD if convertible")
    applies_to: str = Field(..., description="What losses/claims the cap applies to")
    exceptions: list[str] = Field(default_factory=list, description="Carve-outs from the cap")
    confidence: ClampedConfidence = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")

    @field_validator("cap_type", mode="before")
    @classmethod
    def validate_cap_type(cls, v: str) -> str:
        """Normalize and validate cap type."""
        v = v.lower().strip().replace(" ", "_")
        valid_types = {"liability_cap", "indemnity_cap", "damages_limitation", "deductible"}
        if v not in valid_types:
            # Best-effort mapping
            if "liability" in v:
                return "liability_cap"
            elif "indemnit" in v:
                return "indemnity_cap"
            elif "damage" in v:
                return "damages_limitation"
            elif "deduct" in v or "basket" in v:
                return "deductible"
            return "liability_cap"  # Default
        return v


class CapLimitationOutput(BaseModel):
    """Output structure from the Cap & Limitation Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    caps: list[Cap] = Field(default_factory=list, description="Extracted caps")
    no_cap_found: bool = Field(default=False, description="True if no caps exist in clause")


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: CapLimitationOutput | None
    success: bool
    model: str
    input_tokens: int
    output_tokens: int
    execution_time_ms: int
    cost_usd: float
    error_message: str | None = None
    retry_count: int = 0
    raw_response: str | None = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay_seconds: float = 1.0
    retryable_errors: list[str] = field(default_factory=lambda: ["timeout", "rate_limit", "server_error"])


class CapLimitationAgent:
    """Stateless agent for extracting caps and limitations from contract clauses.

    This agent uses GPT-5 Nano with deterministic settings (seed=42) to ensure
    consistent outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = CapLimitationAgent()
        input_data = CapLimitationInput(
            clause_id="clause_001",
            clause_text="The Seller's liability shall not exceed $5M...",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware"
            )
        )
        result = agent.extract(input_data)
    """

    # GPT-5 Nano pricing (per 1M tokens)
    INPUT_TOKEN_RATE = 0.05  # $0.05 per 1M input tokens
    OUTPUT_TOKEN_RATE = 0.4  # $0.4 per 1M output tokens
    MODEL_NAME = "gpt-5-nano"
    MAX_COMPLETION_TOKENS = 800
    SEED = 42  # Fixed seed for deterministic outputs

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Cap & Limitation Agent.

        Args:
            cost_tracker: Optional cost tracker for logging. Creates new if None.
            retry_config: Optional retry configuration.
        """
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.cap_limitation_agent")
        self.retry_config = retry_config or RetryConfig()

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client (lazy initialization)."""
        if self._client is None:
            settings = get_settings()
            if not settings.openai_api_key or settings.openai_api_key in ("", "your_openai_api_key_here"):
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def _format_prompt(self, input_data: CapLimitationInput) -> str:
        """Format the user prompt for the LLM.

        Args:
            input_data: The input clause data.

        Returns:
            Formatted prompt string.
        """
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"

        return CAP_LIMITATION_USER_PROMPT_TEMPLATE.format(
            clause_id=input_data.clause_id,
            clause_text=input_data.clause_text,
            contract_type=input_data.metadata.contract_type or "Not specified",
            parties=parties_str,
            jurisdiction=input_data.metadata.jurisdiction or "Not specified",
        )

    def _parse_json_response(self, text: str, clause_id: str) -> CapLimitationOutput:
        """Parse JSON from LLM response with repair strategies.

        Args:
            text: Raw response text from LLM.
            clause_id: Clause ID to use if parsing fails.

        Returns:
            Parsed CapLimitationOutput.

        Raises:
            ValueError: If JSON cannot be parsed after all repair attempts.
        """
        # Strategy 1: Extract from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        text = text.strip()

        # Strategy 2: Try direct parsing
        try:
            data = json.loads(text)
            return self._validate_and_build_output(data, clause_id)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract JSON object with regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_build_output(data, clause_id)
            except json.JSONDecodeError:
                text = json_match.group()

        # Strategy 4: Repair common issues
        repaired = text
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)  # Remove trailing commas
        repaired = re.sub(r'"\s*"', '", "', repaired)  # Fix missing commas in arrays

        try:
            data = json.loads(repaired)
            return self._validate_and_build_output(data, clause_id)
        except json.JSONDecodeError:
            pass

        # Strategy 5: Extract individual fields
        result = {
            "clause_id": clause_id,
            "caps": [],
            "no_cap_found": True,
        }

        # Try to extract caps array
        caps_match = re.search(r'"caps"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if caps_match:
            caps_text = caps_match.group(1)
            cap_objects = re.findall(r'\{[^{}]+\}', caps_text)
            caps = []
            for obj in cap_objects:
                try:
                    parsed = json.loads(obj)
                    caps.append(parsed)
                except json.JSONDecodeError:
                    continue
            if caps:
                result["caps"] = caps
                result["no_cap_found"] = False

        return self._validate_and_build_output(result, clause_id)

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> CapLimitationOutput:
        """Validate and build output from parsed data.

        Args:
            data: Parsed JSON data.
            clause_id: Clause ID to use.

        Returns:
            Validated CapLimitationOutput.
        """
        # Ensure clause_id is set
        data["clause_id"] = data.get("clause_id", clause_id)

        # Parse caps
        caps = []
        raw_caps = data.get("caps", [])

        for raw_cap in raw_caps:
            try:
                # Parse cap amount to get numeric value if not provided
                cap_amount = raw_cap.get("cap_amount", "")
                cap_amount_numeric = raw_cap.get("cap_amount_numeric")
                
                # Try to parse numeric if not provided
                if cap_amount_numeric is None and cap_amount:
                    _, parsed_numeric = parse_cap_amount(cap_amount)
                    cap_amount_numeric = parsed_numeric

                cap = Cap(
                    cap_type=raw_cap.get("cap_type", "liability_cap"),
                    cap_amount=cap_amount,
                    cap_amount_numeric=cap_amount_numeric,
                    applies_to=raw_cap.get("applies_to", ""),
                    exceptions=raw_cap.get("exceptions", []),
                    confidence=float(raw_cap.get("confidence", 0.5)),
                )
                caps.append(cap)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse cap: {e}")
                continue

        no_cap_found = data.get("no_cap_found", len(caps) == 0)

        return CapLimitationOutput(
            clause_id=data["clause_id"],
            caps=caps,
            no_cap_found=no_cap_found,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error should trigger a retry.

        Args:
            error: The exception that occurred.

        Returns:
            True if the error is retryable.
        """
        error_str = str(error).lower()

        # Check for rate limiting
        if "rate" in error_str and "limit" in error_str:
            return True

        # Check for timeout
        if "timeout" in error_str:
            return True

        # Check for server errors
        if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
            return True

        # Check for connection errors
        if "connection" in error_str:
            return True

        return False

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost in USD.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        input_cost = (input_tokens * self.INPUT_TOKEN_RATE) / 1_000_000
        output_cost = (output_tokens * self.OUTPUT_TOKEN_RATE) / 1_000_000
        return input_cost + output_cost

    def extract(self, input_data: CapLimitationInput) -> AgentExecutionResult:
        """Extract caps and limitations from a clause (synchronous).

        This is a stateless operation - each call is independent with no
        conversation history.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with extracted caps and metadata.
        """
        start_time = time.time()
        retry_count = 0
        last_error: str | None = None

        while retry_count <= self.retry_config.max_retries:
            try:
                # Format prompts
                user_prompt = self._format_prompt(input_data)

                # Call GPT-5 Nano with EXACT configuration
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": CAP_LIMITATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                    seed=self.SEED,  # Fixed seed for reproducible outputs (CRITICAL for voting)
                    extra_body={
                        "reasoning_effort": "low",  # Optimized for focused micro-tasks
                        "verbosity": "low",  # Keep responses concise
                    },
                )

                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Parse response
                raw_response = response.choices[0].message.content or "{}"
                output = self._parse_json_response(raw_response, input_data.clause_id)

                # Calculate metrics
                execution_time_ms = int((time.time() - start_time) * 1000)
                cost_usd = self._calculate_cost(input_tokens, output_tokens)

                # Log execution
                avg_confidence = (
                    sum(cap.confidence for cap in output.caps) / len(output.caps)
                    if output.caps
                    else 0.0
                )
                total_cap_value = sum(
                    cap.cap_amount_numeric for cap in output.caps
                    if cap.cap_amount_numeric is not None
                )
                log = self.cost_tracker.create_log(
                    agent_type="CAP_LIMITATION_EXTRACTION",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": round(avg_confidence, 2),
                        "caps_found": len(output.caps),
                        "total_cap_value_usd": round(total_cap_value, 2) if total_cap_value else None,
                    },
                    retry_count=retry_count,
                )
                self.cost_tracker.log_execution(log)

                return AgentExecutionResult(
                    output=output,
                    success=True,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    cost_usd=cost_usd,
                    retry_count=retry_count,
                    raw_response=raw_response,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Extraction attempt {retry_count + 1} failed: {last_error}")

                if not self._is_retryable_error(e) or retry_count >= self.retry_config.max_retries:
                    break

                # Exponential backoff
                delay = self.retry_config.initial_delay_seconds * (
                    self.retry_config.backoff_factor ** retry_count
                )
                time.sleep(delay)
                retry_count += 1

        # All retries exhausted or non-retryable error
        execution_time_ms = int((time.time() - start_time) * 1000)

        log = self.cost_tracker.create_log(
            agent_type="CAP_LIMITATION_EXTRACTION",
            clause_id=input_data.clause_id,
            model=self.MODEL_NAME,
            input_tokens=0,
            output_tokens=0,
            execution_time_ms=execution_time_ms,
            status=ExecutionStatus.FAILURE,
            error_message=last_error,
            retry_count=retry_count,
        )
        self.cost_tracker.log_execution(log)

        return AgentExecutionResult(
            output=None,
            success=False,
            model=self.MODEL_NAME,
            input_tokens=0,
            output_tokens=0,
            execution_time_ms=execution_time_ms,
            cost_usd=0.0,
            error_message=last_error,
            retry_count=retry_count,
        )

    async def extract_async(self, input_data: CapLimitationInput) -> AgentExecutionResult:
        """Extract caps from a clause (asynchronous).

        This wraps the synchronous extract method for async contexts.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with extracted caps and metadata.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, input_data)

    async def extract_batch(
        self,
        inputs: list[CapLimitationInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Extract caps from multiple clauses in parallel.

        Args:
            inputs: List of clause inputs to process.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of results in same order as inputs.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_extract(input_data: CapLimitationInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.extract_async(input_data)

        tasks = [bounded_extract(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class CapLimitationRepository:
    """Repository for storing and retrieving cap extraction results from PostgreSQL."""

    def __init__(self, pool: Any) -> None:
        """Initialize with a database connection pool.

        Args:
            pool: asyncpg connection pool.
        """
        self.pool = pool

    async def ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cap_limitation_extractions (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    caps JSONB NOT NULL,
                    no_cap_found BOOLEAN DEFAULT FALSE,
                    total_cap_value_usd DECIMAL(15, 2),
                    model VARCHAR(50) NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    cost_usd DECIMAL(10, 6) NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(clause_id)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cap_limitation_extractions_clause_id
                ON cap_limitation_extractions(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cap_limitation_extractions_document_id
                ON cap_limitation_extractions(document_id)
            """)

    async def save_result(
        self,
        clause_id: str,
        result: AgentExecutionResult,
        document_id: str | None = None,
    ) -> int:
        """Save an extraction result to the database.

        Args:
            clause_id: ID of the clause.
            result: The extraction result.
            document_id: Optional document ID.

        Returns:
            ID of the inserted row.
        """
        caps_json = (
            [cap.model_dump() for cap in result.output.caps]
            if result.output
            else []
        )
        no_cap_found = result.output.no_cap_found if result.output else True
        
        # Calculate total cap value
        total_cap_value = None
        if result.output and result.output.caps:
            numeric_caps = [
                cap.cap_amount_numeric for cap in result.output.caps
                if cap.cap_amount_numeric is not None
            ]
            if numeric_caps:
                total_cap_value = sum(numeric_caps)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO cap_limitation_extractions (
                    clause_id, document_id, caps, no_cap_found, total_cap_value_usd,
                    model, input_tokens, output_tokens, execution_time_ms,
                    cost_usd, success, error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (clause_id) DO UPDATE SET
                    caps = EXCLUDED.caps,
                    no_cap_found = EXCLUDED.no_cap_found,
                    total_cap_value_usd = EXCLUDED.total_cap_value_usd,
                    model = EXCLUDED.model,
                    input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    cost_usd = EXCLUDED.cost_usd,
                    success = EXCLUDED.success,
                    error_message = EXCLUDED.error_message,
                    retry_count = EXCLUDED.retry_count,
                    created_at = NOW()
                RETURNING id
                """,
                clause_id,
                document_id,
                json.dumps(caps_json),
                no_cap_found,
                total_cap_value,
                result.model,
                result.input_tokens,
                result.output_tokens,
                result.execution_time_ms,
                result.cost_usd,
                result.success,
                result.error_message,
                result.retry_count,
            )
            return row["id"]

    async def get_result(self, clause_id: str) -> dict[str, Any] | None:
        """Retrieve an extraction result by clause ID.

        Args:
            clause_id: ID of the clause.

        Returns:
            Result dict or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM cap_limitation_extractions WHERE clause_id = $1
                """,
                clause_id,
            )
            if row:
                return dict(row)
            return None

    async def get_results_by_document(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all extraction results for a document.

        Args:
            document_id: ID of the document.

        Returns:
            List of result dicts.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM cap_limitation_extractions
                WHERE document_id = $1
                ORDER BY created_at
                """,
                document_id,
            )
            return [dict(row) for row in rows]

    async def get_cap_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get financial summary of all caps for a document or globally.

        Args:
            document_id: Optional document ID to filter by.

        Returns:
            Summary dict with cap metrics.
        """
        query = """
            SELECT
                COUNT(*) as total_clauses,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN NOT no_cap_found THEN 1 ELSE 0 END) as clauses_with_caps,
                SUM(total_cap_value_usd) as total_identified_cap_value,
                AVG(total_cap_value_usd) FILTER (WHERE total_cap_value_usd IS NOT NULL) as avg_cap_value,
                MIN(total_cap_value_usd) FILTER (WHERE total_cap_value_usd IS NOT NULL) as min_cap_value,
                MAX(total_cap_value_usd) FILTER (WHERE total_cap_value_usd IS NOT NULL) as max_cap_value,
                SUM(cost_usd) as total_cost_usd,
                AVG(cost_usd) as avg_cost_per_extraction,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM cap_limitation_extractions
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}


async def fetch_liability_clauses(pool: Any, document_id: str) -> list[CapLimitationInput]:
    """Fetch liability-related clauses from the database for cap extraction.

    This function queries the decomposition engine output to get clauses
    that are likely to contain caps and limitations.

    Args:
        pool: asyncpg connection pool.
        document_id: ID of the document to fetch clauses for.

    Returns:
        List of CapLimitationInput objects.
    """
    async with pool.acquire() as conn:
        # Check if decomposed_clauses table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'decomposed_clauses'
            )
        """)

        if not table_exists:
            logger.warning("decomposed_clauses table does not exist")
            return []

        # Fetch liability-related clauses
        rows = await conn.fetch(
            """
            SELECT
                clause_id,
                clause_text,
                clause_type,
                contract_type,
                parties,
                jurisdiction,
                execution_date
            FROM decomposed_clauses
            WHERE document_id = $1
              AND (
                  clause_type ILIKE '%liability%'
                  OR clause_type ILIKE '%indemn%'
                  OR clause_type ILIKE '%damage%'
                  OR clause_type ILIKE '%limit%'
                  OR clause_text ILIKE '%shall not exceed%'
                  OR clause_text ILIKE '%maximum liability%'
                  OR clause_text ILIKE '%capped at%'
                  OR clause_text ILIKE '%basket%'
                  OR clause_text ILIKE '%deductible%'
              )
            ORDER BY sequence_number
            """,
            document_id,
        )

        inputs = []
        for row in rows:
            parties = row["parties"] if row["parties"] else []
            if isinstance(parties, str):
                parties = json.loads(parties) if parties.startswith("[") else [parties]

            inputs.append(
                CapLimitationInput(
                    clause_id=row["clause_id"],
                    clause_text=row["clause_text"],
                    clause_type=row["clause_type"] or "",
                    metadata=ClauseMetadata(
                        contract_type=row["contract_type"] or "",
                        parties=parties,
                        jurisdiction=row["jurisdiction"] or "",
                        execution_date=row["execution_date"],
                    ),
                )
            )

        return inputs
