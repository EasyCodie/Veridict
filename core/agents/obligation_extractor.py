"""Obligation Extractor Agent - Priority 1 Worker in Micro-Agent Factory.

This agent extracts who must do what, under what conditions, from individual
contract clauses. It uses GPT-5 Nano with deterministic settings for
consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
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
from pydantic import BaseModel, Field, BeforeValidator
from typing_extensions import Annotated

from app.config import get_settings
from core.cost_tracker import CostTracker, ExecutionStatus


def clamp_confidence(v: float) -> float:
    """Clamp confidence value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


# Type alias for clamped confidence values
ClampedConfidence = Annotated[float, BeforeValidator(clamp_confidence)]


logger = logging.getLogger("veridict.obligation_extractor")


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


class ObligationExtractorInput(BaseModel):
    """Input contract for the Obligation Extractor Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    clause_type: str = Field(default="", description="Type of clause (optional)")
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class Obligation(BaseModel):
    """A single extracted obligation."""
    obligor: str = Field(..., description="Who must perform the obligation")
    obligation_description: str = Field(..., description="What must be done")
    trigger_condition: str = Field(..., description="When/if the obligation applies")
    deadline: str | None = Field(default=None, description="Deadline for performance")
    consequences_of_breach: str | None = Field(default=None, description="Consequences of non-performance")
    confidence: ClampedConfidence = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")


class ObligationExtractorOutput(BaseModel):
    """Output structure from the Obligation Extractor Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    obligations: list[Obligation] = Field(default_factory=list, description="Extracted obligations")
    no_obligations_found: bool = Field(default=False, description="True if no obligations exist")


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: ObligationExtractorOutput | None
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


class ObligationExtractorAgent:
    """Stateless agent for extracting obligations from contract clauses.

    This agent uses GPT-5 Nano with deterministic settings (seed=42) to ensure
    consistent outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = ObligationExtractorAgent()
        input_data = ObligationExtractorInput(
            clause_id="clause_001",
            clause_text="The Seller shall deliver...",
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
        """Initialize the Obligation Extractor Agent.

        Args:
            cost_tracker: Optional cost tracker for logging. Creates new if None.
            retry_config: Optional retry configuration.
        """
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.obligation_extractor")
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

    def _format_prompt(self, input_data: ObligationExtractorInput) -> str:
        """Format the user prompt for the LLM.

        Args:
            input_data: The input clause data.

        Returns:
            Formatted prompt string.
        """
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"

        return f"""You are a legal obligation extractor analyzing a single contract clause.

CLAUSE TEXT:
{input_data.clause_text}

CONTRACT METADATA:
- Type: {input_data.metadata.contract_type or "Not specified"}
- Parties: {parties_str}
- Jurisdiction: {input_data.metadata.jurisdiction or "Not specified"}

TASK:
Extract all obligations (who must do what, under what conditions) from this clause ONLY.

RULES:
1. Extract only what is explicitly stated in this clause.
2. Do NOT infer obligations from other sections.
3. For each obligation, identify:
   - WHO must act (the obligor)
   - WHAT they must do (obligation description)
   - WHEN/IF it applies (trigger condition and deadline)
   - WHAT happens if they don't (consequences)
4. If an obligation is unclear, mark confidence < 0.7.
5. If no obligations exist, return empty obligations array with "no_obligations_found": true.

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{input_data.clause_id}",
  "obligations": [
    {{
      "obligor": "string",
      "obligation_description": "string",
      "trigger_condition": "string",
      "deadline": "string or null",
      "consequences_of_breach": "string or null",
      "confidence": number between 0 and 1
    }}
  ],
  "no_obligations_found": boolean
}}"""

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a legal obligation extractor specializing in contract analysis.

Your role is to extract all obligations from a single contract clause.
You must identify WHO must do WHAT, WHEN, and what happens if they don't.

Key principles:
- Extract only what is explicitly stated in the clause
- Do not infer obligations from context outside the clause
- Mark unclear obligations with confidence < 0.7
- Always provide structured JSON output
- Back findings with exact text from the clause"""

    def _parse_json_response(self, text: str, clause_id: str) -> ObligationExtractorOutput:
        """Parse JSON from LLM response with repair strategies.

        Args:
            text: Raw response text from LLM.
            clause_id: Clause ID to use if parsing fails.

        Returns:
            Parsed ObligationExtractorOutput.

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
            "obligations": [],
            "no_obligations_found": True,
        }

        # Try to extract obligations array
        oblig_match = re.search(r'"obligations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if oblig_match:
            oblig_text = oblig_match.group(1)
            # Try to parse individual obligations
            oblig_objects = re.findall(r'\{[^{}]+\}', oblig_text)
            obligations = []
            for obj in oblig_objects:
                try:
                    parsed = json.loads(obj)
                    obligations.append(parsed)
                except json.JSONDecodeError:
                    continue
            if obligations:
                result["obligations"] = obligations
                result["no_obligations_found"] = False

        return self._validate_and_build_output(result, clause_id)

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> ObligationExtractorOutput:
        """Validate and build output from parsed data.

        Args:
            data: Parsed JSON data.
            clause_id: Clause ID to use.

        Returns:
            Validated ObligationExtractorOutput.
        """
        # Ensure clause_id is set
        data["clause_id"] = data.get("clause_id", clause_id)

        # Parse obligations
        obligations = []
        raw_obligations = data.get("obligations", [])

        for raw_ob in raw_obligations:
            try:
                # Ensure all required fields have defaults
                obligation = Obligation(
                    obligor=raw_ob.get("obligor", "Unknown"),
                    obligation_description=raw_ob.get("obligation_description", ""),
                    trigger_condition=raw_ob.get("trigger_condition", "Not specified"),
                    deadline=raw_ob.get("deadline"),
                    consequences_of_breach=raw_ob.get("consequences_of_breach"),
                    confidence=float(raw_ob.get("confidence", 0.5)),
                )
                obligations.append(obligation)
            except (ValueError, TypeError):
                continue

        no_obligations = data.get("no_obligations_found", len(obligations) == 0)

        return ObligationExtractorOutput(
            clause_id=data["clause_id"],
            obligations=obligations,
            no_obligations_found=no_obligations,
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

    def extract(self, input_data: ObligationExtractorInput) -> AgentExecutionResult:
        """Extract obligations from a clause (synchronous).

        This is a stateless operation - each call is independent with no
        conversation history.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with extracted obligations and metadata.
        """
        start_time = time.time()
        retry_count = 0
        last_error: str | None = None

        while retry_count <= self.retry_config.max_retries:
            try:
                # Format prompts
                system_prompt = self._get_system_prompt()
                user_prompt = self._format_prompt(input_data)

                # Call GPT-5 Nano with EXACT configuration
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
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
                    sum(ob.confidence for ob in output.obligations) / len(output.obligations)
                    if output.obligations
                    else 0.0
                )
                log = self.cost_tracker.create_log(
                    agent_type="OBLIGATION_EXTRACTION",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": round(avg_confidence, 2),
                        "obligations_found": len(output.obligations),
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
            agent_type="OBLIGATION_EXTRACTION",
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

    async def extract_async(self, input_data: ObligationExtractorInput) -> AgentExecutionResult:
        """Extract obligations from a clause (asynchronous).

        This wraps the synchronous extract method for async contexts.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with extracted obligations and metadata.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, input_data)

    async def extract_batch(
        self,
        inputs: list[ObligationExtractorInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Extract obligations from multiple clauses in parallel.

        Args:
            inputs: List of clause inputs to process.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of results in same order as inputs.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_extract(input_data: ObligationExtractorInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.extract_async(input_data)

        tasks = [bounded_extract(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class ObligationExtractorRepository:
    """Repository for storing and retrieving obligation extraction results from PostgreSQL."""

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
                CREATE TABLE IF NOT EXISTS obligation_extractions (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    obligations JSONB NOT NULL,
                    no_obligations_found BOOLEAN DEFAULT FALSE,
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
                CREATE INDEX IF NOT EXISTS idx_obligation_extractions_clause_id
                ON obligation_extractions(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_obligation_extractions_document_id
                ON obligation_extractions(document_id)
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
        obligations_json = (
            [ob.model_dump() for ob in result.output.obligations]
            if result.output
            else []
        )
        no_obligations = result.output.no_obligations_found if result.output else True

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO obligation_extractions (
                    clause_id, document_id, obligations, no_obligations_found,
                    model, input_tokens, output_tokens, execution_time_ms,
                    cost_usd, success, error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (clause_id) DO UPDATE SET
                    obligations = EXCLUDED.obligations,
                    no_obligations_found = EXCLUDED.no_obligations_found,
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
                json.dumps(obligations_json),
                no_obligations,
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
                SELECT * FROM obligation_extractions WHERE clause_id = $1
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
                SELECT * FROM obligation_extractions
                WHERE document_id = $1
                ORDER BY created_at
                """,
                document_id,
            )
            return [dict(row) for row in rows]

    async def get_cost_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get cost summary for extractions.

        Args:
            document_id: Optional document ID to filter by.

        Returns:
            Summary dict with cost metrics.
        """
        query = """
            SELECT
                COUNT(*) as total_extractions,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                SUM(cost_usd) as total_cost_usd,
                AVG(cost_usd) as avg_cost_per_extraction,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM obligation_extractions
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}


async def fetch_decomposed_clauses(pool: Any, document_id: str) -> list[ObligationExtractorInput]:
    """Fetch decomposed clauses from the database for processing.

    This function queries the decomposition engine output to get clauses
    ready for obligation extraction.

    Args:
        pool: asyncpg connection pool.
        document_id: ID of the document to fetch clauses for.

    Returns:
        List of ObligationExtractorInput objects.
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
                ObligationExtractorInput(
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


# Global agent instance (lazy initialization)
_agent: ObligationExtractorAgent | None = None


def get_obligation_extractor_agent() -> ObligationExtractorAgent:
    """Get the global obligation extractor agent instance.

    Returns:
        ObligationExtractorAgent instance.
    """
    global _agent
    if _agent is None:
        _agent = ObligationExtractorAgent()
    return _agent
