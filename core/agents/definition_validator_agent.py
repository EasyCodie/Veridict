"""Definition Validator Agent - Priority 5 Worker in Micro-Agent Factory.

This agent identifies undefined or ambiguously defined terms in contract
clauses and flags them for legal review. It uses GPT-5 Nano with deterministic
settings for consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
- Cross-reference support with definitions section
- Severity classification (critical, high, medium)
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
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, BeforeValidator, field_validator
from typing_extensions import Annotated

from app.config import get_settings
from core.cost_tracker import CostTracker, ExecutionStatus
from core.agents.utils.definition_utilities import (
    DefinitionSeverity,
    extract_defined_terms,
    find_capitalized_terms,
    is_term_defined,
    suggest_definition_location,
    aggregate_undefined_terms,
    normalize_severity,
)
from core.agents.prompts.definition_validator_prompt import (
    DEFINITION_VALIDATOR_SYSTEM_PROMPT,
    DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE,
)


def clamp_confidence(v: float) -> float:
    """Clamp confidence value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


ClampedConfidence = Annotated[float, BeforeValidator(clamp_confidence)]

logger = logging.getLogger("veridict.definition_validator_agent")


class ClauseMetadata(BaseModel):
    """Metadata about the contract containing the clause."""
    contract_type: str = Field(default="", description="Type of contract (e.g., 'SPA', 'NDA')")
    parties: list[str] = Field(default_factory=list, description="Parties to the contract")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    execution_date: str | None = Field(default=None, description="Execution date (ISO8601)")


class DefinitionValidatorInput(BaseModel):
    """Input contract for the Definition Validator Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    clause_type: str = Field(default="", description="Type of clause")
    definitions_section: str | None = Field(default=None, description="Text from Definitions clause")
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class UndefinedTerm(BaseModel):
    """An undefined term identified in a clause."""
    term: str = Field(..., description="The undefined term")
    context_usage: str = Field(default="", description="How the term is used in the clause")
    severity: str = Field(..., description="Severity level")
    likelihood_defined_elsewhere: bool = Field(default=False, description="Likely defined elsewhere")
    search_suggestions: list[str] = Field(default_factory=list, description="Where to look")
    
    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Normalize severity to valid enum."""
        return normalize_severity(v)


class DefinitionValidatorOutput(BaseModel):
    """Output structure from the Definition Validator Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    undefined_terms: list[UndefinedTerm] = Field(default_factory=list, description="Undefined terms")
    all_terms_defined: bool = Field(default=True, description="Whether all terms are defined")
    definition_gaps_summary: str | None = Field(default=None, description="Summary of gaps")
    confidence_score: ClampedConfidence = Field(default=0.8, ge=0.0, le=1.0)


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: DefinitionValidatorOutput | None
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


class DefinitionValidatorAgent:
    """Stateless agent for validating term definitions in contract clauses.

    Uses GPT-5 Nano with deterministic settings (seed=42) for consistent
    outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = DefinitionValidatorAgent()
        input_data = DefinitionValidatorInput(
            clause_id="clause_001",
            clause_text="The Seller represents that no Material Adverse Effect...",
            definitions_section='"Material Adverse Effect" means...',
            metadata=ClauseMetadata(contract_type="SPA")
        )
        result = agent.validate(input_data)
    """

    # GPT-5 Nano pricing (per 1M tokens)
    INPUT_TOKEN_RATE = 0.05
    OUTPUT_TOKEN_RATE = 0.4
    MODEL_NAME = "gpt-5-nano"
    MAX_COMPLETION_TOKENS = 800
    SEED = 42

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Definition Validator Agent."""
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.definition_validator_agent")
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

    def _format_prompt(self, input_data: DefinitionValidatorInput) -> str:
        """Format the user prompt for the LLM."""
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"
        definitions_str = input_data.definitions_section or "Not provided - assume terms may be defined elsewhere"

        return DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE.format(
            clause_id=input_data.clause_id,
            clause_text=input_data.clause_text,
            clause_type=input_data.clause_type or "Not specified",
            definitions_section=definitions_str,
            contract_type=input_data.metadata.contract_type or "Not specified",
            parties=parties_str,
            jurisdiction=input_data.metadata.jurisdiction or "Not specified",
        )

    def _parse_json_response(self, text: str, clause_id: str) -> DefinitionValidatorOutput:
        """Parse JSON from LLM response with repair strategies."""
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
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        repaired = re.sub(r'"\s*"', '", "', repaired)

        try:
            data = json.loads(repaired)
            return self._validate_and_build_output(data, clause_id)
        except json.JSONDecodeError:
            pass

        # Strategy 5: Return default output
        return DefinitionValidatorOutput(
            clause_id=clause_id,
            undefined_terms=[],
            all_terms_defined=True,
            definition_gaps_summary=None,
            confidence_score=0.5,
        )

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> DefinitionValidatorOutput:
        """Validate and build output from parsed data."""
        data["clause_id"] = data.get("clause_id", clause_id)

        undefined_terms = []
        raw_terms = data.get("undefined_terms", [])

        for raw_term in raw_terms:
            try:
                term = UndefinedTerm(
                    term=raw_term.get("term", ""),
                    context_usage=raw_term.get("context_usage", ""),
                    severity=raw_term.get("severity", "medium"),
                    likelihood_defined_elsewhere=raw_term.get("likelihood_defined_elsewhere", False),
                    search_suggestions=raw_term.get("search_suggestions", []),
                )
                if term.term:  # Only add if term is not empty
                    undefined_terms.append(term)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse undefined term: {e}")
                continue

        all_terms_defined = len(undefined_terms) == 0

        return DefinitionValidatorOutput(
            clause_id=data["clause_id"],
            undefined_terms=undefined_terms,
            all_terms_defined=all_terms_defined,
            definition_gaps_summary=data.get("definition_gaps_summary"),
            confidence_score=float(data.get("confidence_score", 0.8)),
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error should trigger a retry."""
        error_str = str(error).lower()
        
        if "rate" in error_str and "limit" in error_str:
            return True
        if "timeout" in error_str:
            return True
        if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
            return True
        if "connection" in error_str:
            return True
        return False

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost in USD."""
        input_cost = (input_tokens * self.INPUT_TOKEN_RATE) / 1_000_000
        output_cost = (output_tokens * self.OUTPUT_TOKEN_RATE) / 1_000_000
        return input_cost + output_cost

    def _log_critical_terms(self, output: DefinitionValidatorOutput) -> None:
        """Log critical undefined terms immediately."""
        for term in output.undefined_terms:
            if term.severity == "critical":
                logger.warning(
                    f"⚠️ CRITICAL UNDEFINED TERM in {output.clause_id}: "
                    f"'{term.term}' - {term.context_usage[:80]}..."
                )

    def validate(self, input_data: DefinitionValidatorInput) -> AgentExecutionResult:
        """Validate term definitions in a clause (synchronous).

        This is a stateless operation - each call is independent.

        Args:
            input_data: The clause data to validate.

        Returns:
            AgentExecutionResult with validation results and metadata.
        """
        start_time = time.time()
        retry_count = 0
        last_error: str | None = None

        while retry_count <= self.retry_config.max_retries:
            try:
                user_prompt = self._format_prompt(input_data)

                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": DEFINITION_VALIDATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                    seed=self.SEED,
                    extra_body={
                        "reasoning_effort": "low",
                        "verbosity": "low",
                    },
                )

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                raw_response = response.choices[0].message.content or "{}"
                output = self._parse_json_response(raw_response, input_data.clause_id)

                execution_time_ms = int((time.time() - start_time) * 1000)
                cost_usd = self._calculate_cost(input_tokens, output_tokens)

                # Log critical undefined terms
                self._log_critical_terms(output)

                # Aggregate metrics
                term_stats = aggregate_undefined_terms(
                    [{"severity": t.severity, "likelihood_defined_elsewhere": t.likelihood_defined_elsewhere} 
                     for t in output.undefined_terms]
                )

                log = self.cost_tracker.create_log(
                    agent_type="DEFINITION_VALIDATION",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": output.confidence_score,
                        "undefined_terms_count": len(output.undefined_terms),
                        "all_terms_defined": output.all_terms_defined,
                        "critical_count": term_stats["by_severity"].get("critical", 0),
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
                logger.warning(f"Definition validation attempt {retry_count + 1} failed: {last_error}")

                if not self._is_retryable_error(e) or retry_count >= self.retry_config.max_retries:
                    break

                delay = self.retry_config.initial_delay_seconds * (
                    self.retry_config.backoff_factor ** retry_count
                )
                time.sleep(delay)
                retry_count += 1

        execution_time_ms = int((time.time() - start_time) * 1000)

        log = self.cost_tracker.create_log(
            agent_type="DEFINITION_VALIDATION",
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

    async def validate_async(self, input_data: DefinitionValidatorInput) -> AgentExecutionResult:
        """Validate term definitions in a clause (asynchronous)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate, input_data)

    async def validate_batch(
        self,
        inputs: list[DefinitionValidatorInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Validate multiple clauses in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_validate(input_data: DefinitionValidatorInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.validate_async(input_data)

        tasks = [bounded_validate(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class DefinitionValidatorRepository:
    """Repository for storing and retrieving definition validation results."""

    def __init__(self, pool: Any) -> None:
        """Initialize with a database connection pool."""
        self.pool = pool

    async def ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS definition_validation_results (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    undefined_terms JSONB NOT NULL,
                    all_terms_defined BOOLEAN NOT NULL,
                    definition_gaps_summary TEXT,
                    undefined_count INTEGER DEFAULT 0,
                    critical_count INTEGER DEFAULT 0,
                    high_count INTEGER DEFAULT 0,
                    confidence_score DECIMAL(5, 4) NOT NULL,
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
                CREATE INDEX IF NOT EXISTS idx_def_validation_clause_id
                ON definition_validation_results(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_def_validation_document_id
                ON definition_validation_results(document_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_def_validation_has_undefined
                ON definition_validation_results(all_terms_defined)
                WHERE all_terms_defined = FALSE
            """)

    async def save_result(
        self,
        clause_id: str,
        result: AgentExecutionResult,
        document_id: str | None = None,
    ) -> int:
        """Save a definition validation result to the database."""
        undefined_terms_json = (
            [t.model_dump() for t in result.output.undefined_terms]
            if result.output
            else []
        )
        all_terms_defined = result.output.all_terms_defined if result.output else True
        definition_gaps_summary = result.output.definition_gaps_summary if result.output else None
        confidence_score = result.output.confidence_score if result.output else 0.0
        
        term_stats = aggregate_undefined_terms(
            [{"severity": t.severity, "likelihood_defined_elsewhere": t.likelihood_defined_elsewhere} 
             for t in (result.output.undefined_terms if result.output else [])]
        )

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO definition_validation_results (
                    clause_id, document_id, undefined_terms, all_terms_defined,
                    definition_gaps_summary, undefined_count, critical_count, high_count,
                    confidence_score, model, input_tokens, output_tokens,
                    execution_time_ms, cost_usd, success, error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (clause_id) DO UPDATE SET
                    undefined_terms = EXCLUDED.undefined_terms,
                    all_terms_defined = EXCLUDED.all_terms_defined,
                    definition_gaps_summary = EXCLUDED.definition_gaps_summary,
                    undefined_count = EXCLUDED.undefined_count,
                    critical_count = EXCLUDED.critical_count,
                    high_count = EXCLUDED.high_count,
                    confidence_score = EXCLUDED.confidence_score,
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
                json.dumps(undefined_terms_json),
                all_terms_defined,
                definition_gaps_summary,
                term_stats["total_undefined"],
                term_stats["by_severity"].get("critical", 0),
                term_stats["by_severity"].get("high", 0),
                confidence_score,
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
        """Retrieve a definition validation result by clause ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM definition_validation_results WHERE clause_id = $1""",
                clause_id,
            )
            if row:
                return dict(row)
            return None

    async def get_clauses_with_undefined_terms(
        self, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get clauses that have undefined terms."""
        query = """
            SELECT clause_id, undefined_count, critical_count, high_count,
                   definition_gaps_summary, undefined_terms
            FROM definition_validation_results
            WHERE all_terms_defined = FALSE
        """
        params: list[Any] = []

        if document_id:
            query += " AND document_id = $1"
            params.append(document_id)

        query += " ORDER BY critical_count DESC, undefined_count DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def get_validation_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get definition validation summary for a document or globally."""
        query = """
            SELECT
                COUNT(*) as total_clauses,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN all_terms_defined THEN 1 ELSE 0 END) as fully_defined_clauses,
                SUM(CASE WHEN NOT all_terms_defined THEN 1 ELSE 0 END) as clauses_with_gaps,
                SUM(undefined_count) as total_undefined_terms,
                SUM(critical_count) as total_critical_undefined,
                SUM(high_count) as total_high_undefined,
                AVG(confidence_score) as avg_confidence,
                SUM(cost_usd) as total_cost_usd,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM definition_validation_results
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}


async def fetch_definitions_section(pool: Any, document_id: str) -> str | None:
    """Fetch the definitions section for a document.

    Args:
        pool: asyncpg connection pool.
        document_id: ID of the document.

    Returns:
        Text of definitions section, or None if not found.
    """
    async with pool.acquire() as conn:
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'decomposed_clauses'
            )
        """)

        if not table_exists:
            return None

        row = await conn.fetchrow(
            """
            SELECT clause_text
            FROM decomposed_clauses
            WHERE document_id = $1
              AND (
                  clause_type ILIKE '%definition%'
                  OR clause_text ILIKE '%"%" means%'
                  OR clause_text ILIKE '%"%" shall mean%'
              )
            ORDER BY sequence_number
            LIMIT 1
            """,
            document_id,
        )

        if row:
            return row["clause_text"]
        return None
