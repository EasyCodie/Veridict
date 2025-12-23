"""Comparison Agent - Priority 6 Worker in Micro-Agent Factory.

This agent compares clauses against market-standard language and identifies
deviations that may be disadvantageous. It uses GPT-5 Nano with deterministic
settings for consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
- Market standard comparison
- Deviation taxonomy with 5 types
- Alignment scoring (0-1)
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
from core.agents.utils.deviation_taxonomy import (
    DeviationType,
    Severity,
    OverallAssessment,
    calculate_alignment_score,
    determine_overall_assessment,
    aggregate_deviations,
    normalize_deviation_type,
    normalize_severity,
)
from core.agents.utils.reference_standards import (
    get_reference_standard,
    format_reference_standards_for_prompt,
)
from core.agents.prompts.comparison_agent_prompt import (
    COMPARISON_AGENT_SYSTEM_PROMPT,
    COMPARISON_AGENT_USER_PROMPT_TEMPLATE,
)


def clamp_score(v: float) -> float:
    """Clamp score value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


ClampedScore = Annotated[float, BeforeValidator(clamp_score)]

logger = logging.getLogger("veridict.comparison_agent")


class ReferenceStandardInput(BaseModel):
    """A reference standard provided in the input."""
    standard_type: str = Field(..., description="Type of standard (e.g., 'market_standard_2023')")
    standard_text: str = Field(..., description="Market standard language")
    prevalence: str = Field(default="", description="Prevalence in market (e.g., '85% of SPAs')")


class ClauseMetadata(BaseModel):
    """Metadata about the contract containing the clause."""
    contract_type: str = Field(default="", description="Type of contract (e.g., 'SPA', 'NDA')")
    parties: list[str] = Field(default_factory=list, description="Parties to the contract")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    execution_date: str | None = Field(default=None, description="Execution date (ISO8601)")


class ComparisonInput(BaseModel):
    """Input contract for the Comparison Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    clause_type: str = Field(default="", description="Type of clause")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    contract_type: str = Field(default="", description="Type of contract")
    reference_standards: list[ReferenceStandardInput] = Field(default_factory=list)
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class Deviation(BaseModel):
    """A deviation from market standard language."""
    deviation_type: str = Field(..., description="Type of deviation")
    standard_language: str = Field(default="", description="Quote from reference standard")
    actual_language: str = Field(default="", description="Quote from clause")
    impact: str = Field(default="", description="Who benefits or loses")
    market_prevalence: str = Field(default="", description="How common this deviation is")
    severity: str = Field(..., description="Severity level")
    remediation_suggestion: str = Field(default="", description="How to align with market")
    
    @field_validator("deviation_type", mode="before")
    @classmethod
    def validate_deviation_type(cls, v: str) -> str:
        """Normalize deviation type to valid enum."""
        return normalize_deviation_type(v)
    
    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Normalize severity to valid enum."""
        return normalize_severity(v)


class ComparisonOutput(BaseModel):
    """Output structure from the Comparison Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    deviations_from_standard: list[Deviation] = Field(default_factory=list)
    alignment_score: ClampedScore = Field(default=1.0, ge=0.0, le=1.0)
    overall_assessment: str = Field(default="market_standard")
    confidence_score: ClampedScore = Field(default=0.8, ge=0.0, le=1.0)
    comparison_notes: str | None = Field(default=None)


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: ComparisonOutput | None
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


class ComparisonAgent:
    """Stateless agent for comparing clauses against market standards.

    Uses GPT-5 Nano with deterministic settings (seed=42) for consistent
    outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = ComparisonAgent()
        input_data = ComparisonInput(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify...",
            clause_type="Indemnity",
            reference_standards=[
                ReferenceStandardInput(
                    standard_type="market_standard_2023",
                    standard_text="...",
                    prevalence="85% of M&A transactions"
                )
            ]
        )
        result = agent.compare(input_data)
    """

    # GPT-5 Nano pricing (per 1M tokens)
    INPUT_TOKEN_RATE = 0.05
    OUTPUT_TOKEN_RATE = 0.4
    MODEL_NAME = "gpt-5-nano"
    MAX_COMPLETION_TOKENS = 1000
    SEED = 42

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Comparison Agent."""
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.comparison_agent")
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

    def _format_reference_standards(self, input_data: ComparisonInput) -> str:
        """Format reference standards for the prompt."""
        if input_data.reference_standards:
            parts = []
            for i, std in enumerate(input_data.reference_standards, 1):
                parts.append(
                    f"STANDARD {i} ({std.standard_type}):\n"
                    f"Prevalence: {std.prevalence}\n"
                    f"Language: {std.standard_text}"
                )
            return "\n\n".join(parts)
        
        # Fall back to built-in standards
        standards = get_reference_standard(input_data.clause_type, input_data.jurisdiction)
        return format_reference_standards_for_prompt(standards)

    def _format_prompt(self, input_data: ComparisonInput) -> str:
        """Format the user prompt for the LLM."""
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"
        standards_str = self._format_reference_standards(input_data)

        return COMPARISON_AGENT_USER_PROMPT_TEMPLATE.format(
            clause_id=input_data.clause_id,
            clause_text=input_data.clause_text,
            clause_type=input_data.clause_type or "Not specified",
            reference_standards=standards_str,
            contract_type=input_data.contract_type or input_data.metadata.contract_type or "Not specified",
            parties=parties_str,
            jurisdiction=input_data.jurisdiction or input_data.metadata.jurisdiction or "Not specified",
        )

    def _parse_json_response(self, text: str, clause_id: str) -> ComparisonOutput:
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

        # Strategy 5: Return default output (market standard)
        return ComparisonOutput(
            clause_id=clause_id,
            deviations_from_standard=[],
            alignment_score=1.0,
            overall_assessment="market_standard",
            confidence_score=0.5,
        )

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> ComparisonOutput:
        """Validate and build output from parsed data."""
        data["clause_id"] = data.get("clause_id", clause_id)

        deviations = []
        raw_deviations = data.get("deviations_from_standard", [])

        for raw_dev in raw_deviations:
            try:
                deviation = Deviation(
                    deviation_type=raw_dev.get("deviation_type", "non_standard"),
                    standard_language=raw_dev.get("standard_language", ""),
                    actual_language=raw_dev.get("actual_language", ""),
                    impact=raw_dev.get("impact", ""),
                    market_prevalence=raw_dev.get("market_prevalence", ""),
                    severity=raw_dev.get("severity", "medium"),
                    remediation_suggestion=raw_dev.get("remediation_suggestion", ""),
                )
                deviations.append(deviation)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse deviation: {e}")
                continue

        # Calculate alignment score from deviations
        if deviations:
            deviation_dicts = [{"severity": d.severity} for d in deviations]
            alignment_score = calculate_alignment_score(deviation_dicts)
        else:
            alignment_score = float(data.get("alignment_score", 1.0))

        overall_assessment = determine_overall_assessment(alignment_score).value

        return ComparisonOutput(
            clause_id=data["clause_id"],
            deviations_from_standard=deviations,
            alignment_score=alignment_score,
            overall_assessment=overall_assessment,
            confidence_score=float(data.get("confidence_score", 0.8)),
            comparison_notes=data.get("comparison_notes"),
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

    def _log_significant_deviations(self, output: ComparisonOutput) -> None:
        """Log critical and high severity deviations immediately."""
        for deviation in output.deviations_from_standard:
            if deviation.severity in ("critical", "high"):
                logger.warning(
                    f"⚠️ {deviation.severity.upper()} DEVIATION in {output.clause_id}: "
                    f"{deviation.deviation_type} - {deviation.impact[:80]}..."
                )

    def compare(self, input_data: ComparisonInput) -> AgentExecutionResult:
        """Compare a clause against market standards (synchronous).

        This is a stateless operation - each call is independent.

        Args:
            input_data: The clause data to compare.

        Returns:
            AgentExecutionResult with comparison results and metadata.
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
                        {"role": "system", "content": COMPARISON_AGENT_SYSTEM_PROMPT},
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

                # Log significant deviations
                self._log_significant_deviations(output)

                # Aggregate metrics
                deviation_stats = aggregate_deviations(
                    [{"severity": d.severity, "deviation_type": d.deviation_type} 
                     for d in output.deviations_from_standard]
                )

                log = self.cost_tracker.create_log(
                    agent_type="COMPARISON",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": output.confidence_score,
                        "deviation_count": len(output.deviations_from_standard),
                        "alignment_score": output.alignment_score,
                        "overall_assessment": output.overall_assessment,
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
                logger.warning(f"Comparison attempt {retry_count + 1} failed: {last_error}")

                if not self._is_retryable_error(e) or retry_count >= self.retry_config.max_retries:
                    break

                delay = self.retry_config.initial_delay_seconds * (
                    self.retry_config.backoff_factor ** retry_count
                )
                time.sleep(delay)
                retry_count += 1

        execution_time_ms = int((time.time() - start_time) * 1000)

        log = self.cost_tracker.create_log(
            agent_type="COMPARISON",
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

    async def compare_async(self, input_data: ComparisonInput) -> AgentExecutionResult:
        """Compare a clause against market standards (asynchronous)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compare, input_data)

    async def compare_batch(
        self,
        inputs: list[ComparisonInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Compare multiple clauses in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_compare(input_data: ComparisonInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.compare_async(input_data)

        tasks = [bounded_compare(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class ComparisonRepository:
    """Repository for storing and retrieving comparison results."""

    def __init__(self, pool: Any) -> None:
        """Initialize with a database connection pool."""
        self.pool = pool

    async def ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS comparison_results (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    deviations JSONB NOT NULL,
                    deviation_count INTEGER DEFAULT 0,
                    alignment_score DECIMAL(5, 4) NOT NULL,
                    overall_assessment VARCHAR(50) NOT NULL,
                    critical_count INTEGER DEFAULT 0,
                    high_count INTEGER DEFAULT 0,
                    comparison_notes TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_comparison_clause_id
                ON comparison_results(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_document_id
                ON comparison_results(document_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_assessment
                ON comparison_results(overall_assessment)
            """)

    async def save_result(
        self,
        clause_id: str,
        result: AgentExecutionResult,
        document_id: str | None = None,
    ) -> int:
        """Save a comparison result to the database."""
        deviations_json = (
            [d.model_dump() for d in result.output.deviations_from_standard]
            if result.output
            else []
        )
        alignment_score = result.output.alignment_score if result.output else 1.0
        overall_assessment = result.output.overall_assessment if result.output else "market_standard"
        confidence_score = result.output.confidence_score if result.output else 0.0
        comparison_notes = result.output.comparison_notes if result.output else None
        
        deviation_stats = aggregate_deviations(
            [{"severity": d.severity, "deviation_type": d.deviation_type} 
             for d in (result.output.deviations_from_standard if result.output else [])]
        )

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO comparison_results (
                    clause_id, document_id, deviations, deviation_count,
                    alignment_score, overall_assessment, critical_count, high_count,
                    comparison_notes, confidence_score, model, input_tokens,
                    output_tokens, execution_time_ms, cost_usd, success,
                    error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (clause_id) DO UPDATE SET
                    deviations = EXCLUDED.deviations,
                    deviation_count = EXCLUDED.deviation_count,
                    alignment_score = EXCLUDED.alignment_score,
                    overall_assessment = EXCLUDED.overall_assessment,
                    critical_count = EXCLUDED.critical_count,
                    high_count = EXCLUDED.high_count,
                    comparison_notes = EXCLUDED.comparison_notes,
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
                json.dumps(deviations_json),
                deviation_stats["total_deviations"],
                alignment_score,
                overall_assessment,
                deviation_stats["by_severity"].get("critical", 0),
                deviation_stats["by_severity"].get("high", 0),
                comparison_notes,
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
        """Retrieve a comparison result by clause ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM comparison_results WHERE clause_id = $1""",
                clause_id,
            )
            if row:
                return dict(row)
            return None

    async def get_clauses_with_deviations(
        self, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get clauses that have significant deviations."""
        query = """
            SELECT clause_id, deviation_count, alignment_score, overall_assessment,
                   deviations, comparison_notes
            FROM comparison_results
            WHERE overall_assessment IN ('significant_deviations', 'highly_unusual')
        """
        params: list[Any] = []

        if document_id:
            query += " AND document_id = $1"
            params.append(document_id)

        query += " ORDER BY alignment_score ASC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def get_comparison_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get comparison summary for a document or globally."""
        query = """
            SELECT
                COUNT(*) as total_clauses,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN overall_assessment = 'market_standard' THEN 1 ELSE 0 END) as market_standard_count,
                SUM(CASE WHEN overall_assessment = 'minor_deviations' THEN 1 ELSE 0 END) as minor_deviations_count,
                SUM(CASE WHEN overall_assessment = 'significant_deviations' THEN 1 ELSE 0 END) as significant_deviations_count,
                SUM(CASE WHEN overall_assessment = 'highly_unusual' THEN 1 ELSE 0 END) as highly_unusual_count,
                SUM(deviation_count) as total_deviations,
                SUM(critical_count) as total_critical,
                SUM(high_count) as total_high,
                AVG(alignment_score) as avg_alignment_score,
                AVG(confidence_score) as avg_confidence,
                SUM(cost_usd) as total_cost_usd,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM comparison_results
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}
