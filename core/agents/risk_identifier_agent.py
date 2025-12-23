"""Risk Identifier Agent - Priority 3 Worker in Micro-Agent Factory.

This agent identifies potential risks, red flags, and unusual provisions in
contract clauses. It uses GPT-5 Nano with deterministic settings for
consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
- Risk taxonomy with 6 risk types and 4 severity levels
- Comprehensive cost tracking
- Retry logic with exponential backoff
- PostgreSQL integration for fetch/store
- Contract-level risk aggregation
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
from core.agents.utils.risk_taxonomy import (
    RiskType,
    Severity,
    OverallRiskRating,
    calculate_overall_rating,
    aggregate_risks,
    normalize_risk_type,
    normalize_severity,
)
from core.agents.prompts.risk_identifier_prompt import (
    RISK_IDENTIFIER_SYSTEM_PROMPT,
    RISK_IDENTIFIER_USER_PROMPT_TEMPLATE,
)


def clamp_confidence(v: float) -> float:
    """Clamp confidence value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


ClampedConfidence = Annotated[float, BeforeValidator(clamp_confidence)]

logger = logging.getLogger("veridict.risk_identifier_agent")


class ClauseMetadata(BaseModel):
    """Metadata about the contract containing the clause."""
    contract_type: str = Field(default="", description="Type of contract (e.g., 'SPA', 'NDA')")
    parties: list[str] = Field(default_factory=list, description="Parties to the contract")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    execution_date: str | None = Field(default=None, description="Execution date (ISO8601)")


class RiskIdentifierInput(BaseModel):
    """Input contract for the Risk Identifier Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    clause_type: str = Field(default="", description="Type of clause (e.g., 'Indemnity')")
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class Risk(BaseModel):
    """A single identified risk in a clause."""
    risk_type: str = Field(..., description="Type of risk identified")
    risk_description: str = Field(..., description="Explanation of the risk")
    severity: str = Field(..., description="Severity level (critical, high, medium, low)")
    remediation_suggestion: str = Field(default="", description="How to address this risk")
    confidence: ClampedConfidence = Field(..., ge=0.0, le=1.0, description="Confidence score")
    supporting_quote: str = Field(default="", description="Exact quote from clause")
    
    @field_validator("risk_type", mode="before")
    @classmethod
    def validate_risk_type(cls, v: str) -> str:
        """Normalize risk type to valid enum."""
        return normalize_risk_type(v)
    
    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Normalize severity to valid enum."""
        return normalize_severity(v)


class RiskIdentifierOutput(BaseModel):
    """Output structure from the Risk Identifier Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    risks_identified: list[Risk] = Field(default_factory=list, description="Identified risks")
    overall_risk_rating: str = Field(default="clean", description="Overall risk rating")
    
    @field_validator("overall_risk_rating", mode="before")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        """Normalize overall risk rating."""
        v = v.lower().strip().replace(" ", "_").replace("-", "_")
        valid_ratings = {r.value for r in OverallRiskRating}
        if v in valid_ratings:
            return v
        # Map variations
        if "critical" in v:
            return OverallRiskRating.CRITICAL_ISSUES.value
        if "significant" in v or "major" in v:
            return OverallRiskRating.SIGNIFICANT_ISSUES.value
        if "minor" in v or "low" in v:
            return OverallRiskRating.MINOR_ISSUES.value
        return OverallRiskRating.CLEAN.value


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: RiskIdentifierOutput | None
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


class RiskIdentifierAgent:
    """Stateless agent for identifying risks in contract clauses.

    Uses GPT-5 Nano with deterministic settings (seed=42) for consistent
    outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = RiskIdentifierAgent()
        input_data = RiskIdentifierInput(
            clause_id="clause_001",
            clause_text="The Seller shall indemnify without limitation...",
            clause_type="Indemnity",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware"
            )
        )
        result = agent.identify(input_data)
    """

    # GPT-5 Nano pricing (per 1M tokens)
    INPUT_TOKEN_RATE = 0.05
    OUTPUT_TOKEN_RATE = 0.4
    MODEL_NAME = "gpt-5-nano"
    MAX_COMPLETION_TOKENS = 1000  # Higher for detailed risk descriptions
    SEED = 42

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Risk Identifier Agent."""
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.risk_identifier_agent")
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

    def _format_prompt(self, input_data: RiskIdentifierInput) -> str:
        """Format the user prompt for the LLM."""
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"

        return RISK_IDENTIFIER_USER_PROMPT_TEMPLATE.format(
            clause_id=input_data.clause_id,
            clause_text=input_data.clause_text,
            clause_type=input_data.clause_type or "Not specified",
            contract_type=input_data.metadata.contract_type or "Not specified",
            parties=parties_str,
            jurisdiction=input_data.metadata.jurisdiction or "Not specified",
        )

    def _parse_json_response(self, text: str, clause_id: str) -> RiskIdentifierOutput:
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

        # Strategy 5: Extract individual fields
        result = {
            "clause_id": clause_id,
            "risks_identified": [],
            "overall_risk_rating": "clean",
        }

        risks_match = re.search(r'"risks_identified"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if risks_match:
            risks_text = risks_match.group(1)
            risk_objects = re.findall(r'\{[^{}]+\}', risks_text)
            risks = []
            for obj in risk_objects:
                try:
                    parsed = json.loads(obj)
                    risks.append(parsed)
                except json.JSONDecodeError:
                    continue
            if risks:
                result["risks_identified"] = risks
                result["overall_risk_rating"] = calculate_overall_rating(risks).value

        return self._validate_and_build_output(result, clause_id)

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> RiskIdentifierOutput:
        """Validate and build output from parsed data."""
        data["clause_id"] = data.get("clause_id", clause_id)

        risks = []
        raw_risks = data.get("risks_identified", [])

        for raw_risk in raw_risks:
            try:
                risk = Risk(
                    risk_type=raw_risk.get("risk_type", "unusual_language"),
                    risk_description=raw_risk.get("risk_description", ""),
                    severity=raw_risk.get("severity", "medium"),
                    remediation_suggestion=raw_risk.get("remediation_suggestion", ""),
                    confidence=float(raw_risk.get("confidence", 0.5)),
                    supporting_quote=raw_risk.get("supporting_quote", ""),
                )
                risks.append(risk)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse risk: {e}")
                continue

        # Recalculate overall rating based on actual risks
        if risks:
            risk_dicts = [{"severity": r.severity} for r in risks]
            calculated_rating = calculate_overall_rating(risk_dicts)
        else:
            calculated_rating = OverallRiskRating.CLEAN

        return RiskIdentifierOutput(
            clause_id=data["clause_id"],
            risks_identified=risks,
            overall_risk_rating=calculated_rating.value,
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

    def _log_critical_risks(self, output: RiskIdentifierOutput) -> None:
        """Log critical and high severity risks immediately."""
        for risk in output.risks_identified:
            if risk.severity in ("critical", "high"):
                logger.warning(
                    f"🚨 {risk.severity.upper()} RISK in {output.clause_id}: "
                    f"{risk.risk_type} - {risk.risk_description[:100]}..."
                )

    def identify(self, input_data: RiskIdentifierInput) -> AgentExecutionResult:
        """Identify risks in a clause (synchronous).

        This is a stateless operation - each call is independent.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with identified risks and metadata.
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
                        {"role": "system", "content": RISK_IDENTIFIER_SYSTEM_PROMPT},
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

                # Log critical/high risks
                self._log_critical_risks(output)

                # Aggregate metrics
                risk_stats = aggregate_risks(
                    [{"severity": r.severity, "risk_type": r.risk_type} for r in output.risks_identified]
                )
                avg_confidence = (
                    sum(r.confidence for r in output.risks_identified) / len(output.risks_identified)
                    if output.risks_identified
                    else 0.0
                )

                log = self.cost_tracker.create_log(
                    agent_type="RISK_IDENTIFICATION",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": round(avg_confidence, 2),
                        "risks_found": len(output.risks_identified),
                        "overall_rating": output.overall_risk_rating,
                        "risk_score": risk_stats.get("risk_score", 0),
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
                logger.warning(f"Risk identification attempt {retry_count + 1} failed: {last_error}")

                if not self._is_retryable_error(e) or retry_count >= self.retry_config.max_retries:
                    break

                delay = self.retry_config.initial_delay_seconds * (
                    self.retry_config.backoff_factor ** retry_count
                )
                time.sleep(delay)
                retry_count += 1

        execution_time_ms = int((time.time() - start_time) * 1000)

        log = self.cost_tracker.create_log(
            agent_type="RISK_IDENTIFICATION",
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

    async def identify_async(self, input_data: RiskIdentifierInput) -> AgentExecutionResult:
        """Identify risks in a clause (asynchronous)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.identify, input_data)

    async def identify_batch(
        self,
        inputs: list[RiskIdentifierInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Identify risks in multiple clauses in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_identify(input_data: RiskIdentifierInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.identify_async(input_data)

        tasks = [bounded_identify(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class RiskIdentifierRepository:
    """Repository for storing and retrieving risk identification results."""

    def __init__(self, pool: Any) -> None:
        """Initialize with a database connection pool."""
        self.pool = pool

    async def ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_identification_results (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    risks_identified JSONB NOT NULL,
                    overall_risk_rating VARCHAR(50) NOT NULL,
                    risk_count INTEGER DEFAULT 0,
                    risk_score INTEGER DEFAULT 0,
                    critical_count INTEGER DEFAULT 0,
                    high_count INTEGER DEFAULT 0,
                    medium_count INTEGER DEFAULT 0,
                    low_count INTEGER DEFAULT 0,
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
                CREATE INDEX IF NOT EXISTS idx_risk_results_clause_id
                ON risk_identification_results(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_results_document_id
                ON risk_identification_results(document_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_results_rating
                ON risk_identification_results(overall_risk_rating)
            """)

    async def save_result(
        self,
        clause_id: str,
        result: AgentExecutionResult,
        document_id: str | None = None,
    ) -> int:
        """Save a risk identification result to the database."""
        risks_json = (
            [r.model_dump() for r in result.output.risks_identified]
            if result.output
            else []
        )
        overall_rating = result.output.overall_risk_rating if result.output else "clean"
        
        risk_stats = aggregate_risks(
            [{"severity": r.severity, "risk_type": r.risk_type} for r in (result.output.risks_identified if result.output else [])]
        )

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO risk_identification_results (
                    clause_id, document_id, risks_identified, overall_risk_rating,
                    risk_count, risk_score, critical_count, high_count, medium_count, low_count,
                    model, input_tokens, output_tokens, execution_time_ms,
                    cost_usd, success, error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (clause_id) DO UPDATE SET
                    risks_identified = EXCLUDED.risks_identified,
                    overall_risk_rating = EXCLUDED.overall_risk_rating,
                    risk_count = EXCLUDED.risk_count,
                    risk_score = EXCLUDED.risk_score,
                    critical_count = EXCLUDED.critical_count,
                    high_count = EXCLUDED.high_count,
                    medium_count = EXCLUDED.medium_count,
                    low_count = EXCLUDED.low_count,
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
                json.dumps(risks_json),
                overall_rating,
                risk_stats["total_risks"],
                risk_stats["risk_score"],
                risk_stats["by_severity"].get("critical", 0),
                risk_stats["by_severity"].get("high", 0),
                risk_stats["by_severity"].get("medium", 0),
                risk_stats["by_severity"].get("low", 0),
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
        """Retrieve a risk identification result by clause ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM risk_identification_results WHERE clause_id = $1""",
                clause_id,
            )
            if row:
                return dict(row)
            return None

    async def get_results_by_document(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all risk results for a document."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM risk_identification_results
                WHERE document_id = $1
                ORDER BY risk_score DESC, created_at
                """,
                document_id,
            )
            return [dict(row) for row in rows]

    async def get_risk_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get risk summary for a document or globally."""
        query = """
            SELECT
                COUNT(*) as total_clauses,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                SUM(risk_count) as total_risks,
                SUM(critical_count) as total_critical,
                SUM(high_count) as total_high,
                SUM(medium_count) as total_medium,
                SUM(low_count) as total_low,
                SUM(risk_score) as total_risk_score,
                AVG(risk_score) as avg_risk_score,
                SUM(CASE WHEN overall_risk_rating = 'critical_issues' THEN 1 ELSE 0 END) as critical_clauses,
                SUM(CASE WHEN overall_risk_rating = 'significant_issues' THEN 1 ELSE 0 END) as significant_clauses,
                SUM(CASE WHEN overall_risk_rating = 'minor_issues' THEN 1 ELSE 0 END) as minor_clauses,
                SUM(CASE WHEN overall_risk_rating = 'clean' THEN 1 ELSE 0 END) as clean_clauses,
                SUM(cost_usd) as total_cost_usd,
                AVG(cost_usd) as avg_cost_per_clause,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM risk_identification_results
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}

    async def get_highest_risk_clauses(
        self, document_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get clauses with highest risk scores."""
        query = """
            SELECT clause_id, overall_risk_rating, risk_score, risk_count,
                   critical_count, high_count, risks_identified
            FROM risk_identification_results
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        query += " ORDER BY risk_score DESC, critical_count DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]


async def fetch_clauses_for_risk_analysis(pool: Any, document_id: str) -> list[RiskIdentifierInput]:
    """Fetch all clauses from the database for risk analysis.

    Args:
        pool: asyncpg connection pool.
        document_id: ID of the document to fetch clauses for.

    Returns:
        List of RiskIdentifierInput objects.
    """
    async with pool.acquire() as conn:
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
                RiskIdentifierInput(
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
