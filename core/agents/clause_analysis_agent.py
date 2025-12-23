"""Clause Analysis Agent - Priority 4 Worker in Micro-Agent Factory.

This agent extracts and structures key legal findings from individual clauses
in a comprehensive but concise format. It uses GPT-5 Nano with deterministic
settings for consistent outputs suitable for voting.

Key features:
- Stateless design (no conversation history)
- Deterministic outputs (seed=42)
- Finding taxonomy with 7 finding types
- Importance and risk level classification
- Clarification flagging for ambiguous clauses
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
from core.agents.utils.finding_taxonomy import (
    FindingType,
    Importance,
    RiskLevel,
    calculate_risk_level,
    aggregate_findings,
    normalize_finding_type,
    normalize_importance,
    normalize_risk_level,
)
from core.agents.prompts.clause_analysis_prompt import (
    CLAUSE_ANALYSIS_SYSTEM_PROMPT,
    CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE,
)


def clamp_confidence(v: float) -> float:
    """Clamp confidence value to 0-1 range before validation."""
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    return v


ClampedConfidence = Annotated[float, BeforeValidator(clamp_confidence)]

logger = logging.getLogger("veridict.clause_analysis_agent")


class ClauseMetadata(BaseModel):
    """Metadata about the contract containing the clause."""
    contract_type: str = Field(default="", description="Type of contract (e.g., 'SPA', 'NDA')")
    parties: list[str] = Field(default_factory=list, description="Parties to the contract")
    jurisdiction: str = Field(default="", description="Governing jurisdiction")
    execution_date: str | None = Field(default=None, description="Execution date (ISO8601)")


class ClauseAnalysisInput(BaseModel):
    """Input contract for the Clause Analysis Agent."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    clause_type: str = Field(default="Other", description="Type of clause")
    clause_text: str = Field(..., description="Full text of the clause to analyze")
    task_instructions: str = Field(default="", description="Specific instructions for analysis")
    metadata: ClauseMetadata = Field(default_factory=ClauseMetadata)


class Finding(BaseModel):
    """A single finding extracted from a clause."""
    finding_type: str = Field(..., description="Type of finding")
    finding_text: str = Field(..., description="Direct quote or paraphrase of finding")
    importance: str = Field(..., description="Importance level")
    confidence: ClampedConfidence = Field(..., ge=0.0, le=1.0, description="Confidence score")
    supporting_quote: str = Field(default="", description="Exact quote from clause")
    
    @field_validator("finding_type", mode="before")
    @classmethod
    def validate_finding_type(cls, v: str) -> str:
        """Normalize finding type to valid enum."""
        return normalize_finding_type(v)
    
    @field_validator("importance", mode="before")
    @classmethod
    def validate_importance(cls, v: str) -> str:
        """Normalize importance to valid enum."""
        return normalize_importance(v)


class ClauseAnalysisOutput(BaseModel):
    """Output structure from the Clause Analysis Agent."""
    clause_id: str = Field(..., description="ID of the analyzed clause")
    findings: list[Finding] = Field(default_factory=list, description="Extracted findings")
    summary: str = Field(default="", description="1-2 sentence summary of clause effect")
    risk_level: str = Field(default="informational", description="Overall risk level")
    clarification_required: bool = Field(default=False, description="Whether clarification is needed")
    clarification_note: str | None = Field(default=None, description="What needs review")
    execution_timestamp: str = Field(default="", description="ISO8601 execution timestamp")
    model_used: str = Field(default="gpt-5-nano", description="Model used for analysis")
    
    @field_validator("risk_level", mode="before")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        """Normalize risk level to valid enum."""
        return normalize_risk_level(v)


@dataclass
class AgentExecutionResult:
    """Result of an agent execution including metadata."""
    output: ClauseAnalysisOutput | None
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


class ClauseAnalysisAgent:
    """Stateless agent for extracting findings from contract clauses.

    Uses GPT-5 Nano with deterministic settings (seed=42) for consistent
    outputs suitable for the MAKER Framework voting mechanism.

    Example:
        agent = ClauseAnalysisAgent()
        input_data = ClauseAnalysisInput(
            clause_id="clause_001",
            clause_type="Indemnity",
            clause_text="The Seller shall indemnify...",
            task_instructions="Extract indemnification obligations and caps",
            metadata=ClauseMetadata(
                contract_type="SPA",
                parties=["Seller", "Buyer"],
                jurisdiction="Delaware"
            )
        )
        result = agent.analyze(input_data)
    """

    # GPT-5 Nano pricing (per 1M tokens)
    INPUT_TOKEN_RATE = 0.05
    OUTPUT_TOKEN_RATE = 0.4
    MODEL_NAME = "gpt-5-nano"
    MAX_COMPLETION_TOKENS = 1200  # Higher for comprehensive findings
    SEED = 42

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Clause Analysis Agent."""
        self._client: OpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker(logger_name="veridict.clause_analysis_agent")
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

    def _format_prompt(self, input_data: ClauseAnalysisInput) -> str:
        """Format the user prompt for the LLM."""
        parties_str = ", ".join(input_data.metadata.parties) if input_data.metadata.parties else "Not specified"
        
        default_instructions = "Extract all key findings including obligations, caps, exceptions, definitions, triggers, durations, and processes."

        return CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            clause_id=input_data.clause_id,
            clause_text=input_data.clause_text,
            clause_type=input_data.clause_type or "Other",
            task_instructions=input_data.task_instructions or default_instructions,
            contract_type=input_data.metadata.contract_type or "Not specified",
            parties=parties_str,
            jurisdiction=input_data.metadata.jurisdiction or "Not specified",
        )

    def _parse_json_response(self, text: str, clause_id: str) -> ClauseAnalysisOutput:
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
            "findings": [],
            "summary": "",
            "risk_level": "informational",
            "clarification_required": False,
            "clarification_note": None,
        }

        # Try to extract summary
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', text)
        if summary_match:
            result["summary"] = summary_match.group(1)

        # Try to extract findings
        findings_match = re.search(r'"findings"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1)
            finding_objects = re.findall(r'\{[^{}]+\}', findings_text)
            findings = []
            for obj in finding_objects:
                try:
                    parsed = json.loads(obj)
                    findings.append(parsed)
                except json.JSONDecodeError:
                    continue
            if findings:
                result["findings"] = findings
                result["risk_level"] = calculate_risk_level(findings).value

        return self._validate_and_build_output(result, clause_id)

    def _validate_and_build_output(
        self, data: dict[str, Any], clause_id: str
    ) -> ClauseAnalysisOutput:
        """Validate and build output from parsed data."""
        data["clause_id"] = data.get("clause_id", clause_id)

        findings = []
        raw_findings = data.get("findings", [])

        for raw_finding in raw_findings:
            try:
                finding = Finding(
                    finding_type=raw_finding.get("finding_type", "obligation"),
                    finding_text=raw_finding.get("finding_text", ""),
                    importance=raw_finding.get("importance", "medium"),
                    confidence=float(raw_finding.get("confidence", 0.5)),
                    supporting_quote=raw_finding.get("supporting_quote", ""),
                )
                findings.append(finding)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse finding: {e}")
                continue

        # Recalculate risk level based on actual findings
        if findings:
            finding_dicts = [{"importance": f.importance} for f in findings]
            calculated_risk = calculate_risk_level(finding_dicts)
        else:
            calculated_risk = RiskLevel.INFORMATIONAL

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        return ClauseAnalysisOutput(
            clause_id=data["clause_id"],
            findings=findings,
            summary=data.get("summary", ""),
            risk_level=calculated_risk.value,
            clarification_required=data.get("clarification_required", False),
            clarification_note=data.get("clarification_note"),
            execution_timestamp=timestamp,
            model_used=self.MODEL_NAME,
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

    def _log_important_findings(self, output: ClauseAnalysisOutput) -> None:
        """Log critical and high importance findings immediately."""
        for finding in output.findings:
            if finding.importance in ("critical", "high"):
                logger.warning(
                    f"📋 {finding.importance.upper()} FINDING in {output.clause_id}: "
                    f"{finding.finding_type} - {finding.finding_text[:100]}..."
                )
        
        if output.clarification_required:
            logger.info(
                f"🔍 CLARIFICATION REQUIRED for {output.clause_id}: "
                f"{output.clarification_note or 'See clause for details'}"
            )

    def analyze(self, input_data: ClauseAnalysisInput) -> AgentExecutionResult:
        """Analyze a clause and extract findings (synchronous).

        This is a stateless operation - each call is independent.

        Args:
            input_data: The clause data to analyze.

        Returns:
            AgentExecutionResult with extracted findings and metadata.
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
                        {"role": "system", "content": CLAUSE_ANALYSIS_SYSTEM_PROMPT},
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

                # Log important findings and clarifications
                self._log_important_findings(output)

                # Aggregate metrics
                finding_stats = aggregate_findings(
                    [{"importance": f.importance, "finding_type": f.finding_type} for f in output.findings]
                )
                avg_confidence = (
                    sum(f.confidence for f in output.findings) / len(output.findings)
                    if output.findings
                    else 0.0
                )

                log = self.cost_tracker.create_log(
                    agent_type="CLAUSE_ANALYSIS",
                    clause_id=input_data.clause_id,
                    model=self.MODEL_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time_ms=execution_time_ms,
                    status=ExecutionStatus.SUCCESS,
                    extra_data={
                        "confidence": round(avg_confidence, 2),
                        "findings_count": len(output.findings),
                        "risk_level": output.risk_level,
                        "importance_score": finding_stats.get("importance_score", 0),
                        "clarification_required": output.clarification_required,
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
                logger.warning(f"Clause analysis attempt {retry_count + 1} failed: {last_error}")

                if not self._is_retryable_error(e) or retry_count >= self.retry_config.max_retries:
                    break

                delay = self.retry_config.initial_delay_seconds * (
                    self.retry_config.backoff_factor ** retry_count
                )
                time.sleep(delay)
                retry_count += 1

        execution_time_ms = int((time.time() - start_time) * 1000)

        log = self.cost_tracker.create_log(
            agent_type="CLAUSE_ANALYSIS",
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

    async def analyze_async(self, input_data: ClauseAnalysisInput) -> AgentExecutionResult:
        """Analyze a clause and extract findings (asynchronous)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, input_data)

    async def analyze_batch(
        self,
        inputs: list[ClauseAnalysisInput],
        max_concurrent: int = 10,
    ) -> list[AgentExecutionResult]:
        """Analyze multiple clauses in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_analyze(input_data: ClauseAnalysisInput) -> AgentExecutionResult:
            async with semaphore:
                return await self.analyze_async(input_data)

        tasks = [bounded_analyze(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)


class ClauseAnalysisRepository:
    """Repository for storing and retrieving clause analysis results."""

    def __init__(self, pool: Any) -> None:
        """Initialize with a database connection pool."""
        self.pool = pool

    async def ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS clause_analysis_results (
                    id SERIAL PRIMARY KEY,
                    clause_id VARCHAR(255) NOT NULL,
                    document_id VARCHAR(255),
                    findings JSONB NOT NULL,
                    summary TEXT,
                    risk_level VARCHAR(50) NOT NULL,
                    findings_count INTEGER DEFAULT 0,
                    importance_score INTEGER DEFAULT 0,
                    critical_count INTEGER DEFAULT 0,
                    high_count INTEGER DEFAULT 0,
                    clarification_required BOOLEAN DEFAULT FALSE,
                    clarification_note TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_clause_analysis_clause_id
                ON clause_analysis_results(clause_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_clause_analysis_document_id
                ON clause_analysis_results(document_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_clause_analysis_risk_level
                ON clause_analysis_results(risk_level)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_clause_analysis_clarification
                ON clause_analysis_results(clarification_required)
                WHERE clarification_required = TRUE
            """)

    async def save_result(
        self,
        clause_id: str,
        result: AgentExecutionResult,
        document_id: str | None = None,
    ) -> int:
        """Save a clause analysis result to the database."""
        findings_json = (
            [f.model_dump() for f in result.output.findings]
            if result.output
            else []
        )
        summary = result.output.summary if result.output else ""
        risk_level = result.output.risk_level if result.output else "informational"
        clarification_required = result.output.clarification_required if result.output else False
        clarification_note = result.output.clarification_note if result.output else None
        
        finding_stats = aggregate_findings(
            [{"importance": f.importance, "finding_type": f.finding_type} for f in (result.output.findings if result.output else [])]
        )

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO clause_analysis_results (
                    clause_id, document_id, findings, summary, risk_level,
                    findings_count, importance_score, critical_count, high_count,
                    clarification_required, clarification_note,
                    model, input_tokens, output_tokens, execution_time_ms,
                    cost_usd, success, error_message, retry_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                ON CONFLICT (clause_id) DO UPDATE SET
                    findings = EXCLUDED.findings,
                    summary = EXCLUDED.summary,
                    risk_level = EXCLUDED.risk_level,
                    findings_count = EXCLUDED.findings_count,
                    importance_score = EXCLUDED.importance_score,
                    critical_count = EXCLUDED.critical_count,
                    high_count = EXCLUDED.high_count,
                    clarification_required = EXCLUDED.clarification_required,
                    clarification_note = EXCLUDED.clarification_note,
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
                json.dumps(findings_json),
                summary,
                risk_level,
                finding_stats["total_findings"],
                finding_stats["importance_score"],
                finding_stats["by_importance"].get("critical", 0),
                finding_stats["by_importance"].get("high", 0),
                clarification_required,
                clarification_note,
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
        """Retrieve a clause analysis result by clause ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM clause_analysis_results WHERE clause_id = $1""",
                clause_id,
            )
            if row:
                return dict(row)
            return None

    async def get_results_by_document(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all analysis results for a document."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM clause_analysis_results
                WHERE document_id = $1
                ORDER BY importance_score DESC, created_at
                """,
                document_id,
            )
            return [dict(row) for row in rows]

    async def get_clauses_needing_clarification(
        self, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get clauses that require manual clarification."""
        query = """
            SELECT clause_id, summary, risk_level, clarification_note, findings_count
            FROM clause_analysis_results
            WHERE clarification_required = TRUE
        """
        params: list[Any] = []

        if document_id:
            query += " AND document_id = $1"
            params.append(document_id)

        query += " ORDER BY importance_score DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def get_analysis_summary(self, document_id: str | None = None) -> dict[str, Any]:
        """Get analysis summary for a document or globally."""
        query = """
            SELECT
                COUNT(*) as total_clauses,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                SUM(findings_count) as total_findings,
                SUM(critical_count) as total_critical,
                SUM(high_count) as total_high,
                SUM(CASE WHEN clarification_required THEN 1 ELSE 0 END) as clauses_needing_clarification,
                SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) as critical_risk_clauses,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_clauses,
                SUM(CASE WHEN risk_level = 'informational' THEN 1 ELSE 0 END) as informational_clauses,
                SUM(cost_usd) as total_cost_usd,
                AVG(cost_usd) as avg_cost_per_clause,
                AVG(execution_time_ms) as avg_execution_time_ms
            FROM clause_analysis_results
        """
        params: list[Any] = []

        if document_id:
            query += " WHERE document_id = $1"
            params.append(document_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else {}


async def fetch_clauses_for_analysis(pool: Any, document_id: str) -> list[ClauseAnalysisInput]:
    """Fetch all clauses from the database for analysis.

    Args:
        pool: asyncpg connection pool.
        document_id: ID of the document to fetch clauses for.

    Returns:
        List of ClauseAnalysisInput objects.
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
                clause_type,
                clause_text,
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
                ClauseAnalysisInput(
                    clause_id=row["clause_id"],
                    clause_type=row["clause_type"] or "Other",
                    clause_text=row["clause_text"],
                    metadata=ClauseMetadata(
                        contract_type=row["contract_type"] or "",
                        parties=parties,
                        jurisdiction=row["jurisdiction"] or "",
                        execution_date=row["execution_date"],
                    ),
                )
            )

        return inputs
