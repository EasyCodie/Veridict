"""Cost Tracking and Logging Utilities for Micro-Agent Factory.

This module provides comprehensive cost calculation, tracking, and logging
for all agent operations. It supports GPT-5 Nano and other models with
configurable pricing.

Usage:
    from core.cost_tracker import CostTracker, AgentExecutionLog

    tracker = CostTracker()
    log = tracker.create_log(
        agent_type="OBLIGATION_EXTRACTION",
        clause_id="clause_abc123",
        model="gpt-5-nano",
        input_tokens=245,
        output_tokens=186,
        execution_time_ms=1850,
        status="SUCCESS",
        extra_data={"confidence": 0.87, "obligations_found": 3}
    )
    tracker.log_execution(log)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Status of an agent execution."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ModelPricing(Enum):
    """Pricing per model (cost per 1M tokens)."""
    GPT_5_NANO = {"input": 0.05, "output": 0.4}
    GPT_5_MINI = {"input": 0.25, "output": 2.0}
    GPT_4O_MINI = {"input": 0.15, "output": 0.6}
    GPT_4O = {"input": 2.5, "output": 10.0}

    @classmethod
    def get_pricing(cls, model_name: str) -> dict[str, float]:
        """Get pricing for a model by name."""
        model_map = {
            "gpt-5-nano": cls.GPT_5_NANO,
            "gpt-5-mini": cls.GPT_5_MINI,
            "gpt-4o-mini": cls.GPT_4O_MINI,
            "gpt-4o": cls.GPT_4O,
        }
        pricing = model_map.get(model_name.lower(), cls.GPT_5_NANO)
        return pricing.value


@dataclass
class AgentExecutionLog:
    """Log entry for a single agent execution."""
    timestamp: datetime
    agent_type: str
    clause_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    execution_time_ms: int
    cost_usd: float
    status: ExecutionStatus
    extra_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    retry_count: int = 0

    def to_log_string(self) -> str:
        """Format as a standardized log string."""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        extra_parts = " | ".join(f"{k}={v}" for k, v in self.extra_data.items())
        base = (
            f"[{timestamp_str}] {self.agent_type} | {self.clause_id} | "
            f"model={self.model} | input_tokens={self.input_tokens} | "
            f"output_tokens={self.output_tokens} | "
            f"execution_time_ms={self.execution_time_ms} | "
            f"cost_usd={self.cost_usd:.5f} | status={self.status.value}"
        )
        if extra_parts:
            base += f" | {extra_parts}"
        if self.error_message:
            base += f" | error={self.error_message}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_type": self.agent_type,
            "clause_id": self.clause_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "execution_time_ms": self.execution_time_ms,
            "cost_usd": self.cost_usd,
            "status": self.status.value,
            "extra_data": self.extra_data,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


@dataclass
class BatchCostSummary:
    """Summary statistics for a batch of agent executions."""
    batch_id: str
    total_clauses: int
    successful_clauses: int
    failed_clauses: int
    total_cost_usd: float
    avg_cost_per_clause: float
    total_input_tokens: int
    total_output_tokens: int
    avg_execution_time_ms: float
    success_rate_percent: float
    total_execution_time_ms: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_log_string(self) -> str:
        """Format as a standardized summary log string."""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{timestamp_str}] COST_SUMMARY | {self.batch_id} | "
            f"total_clauses={self.total_clauses} | "
            f"total_cost_usd={self.total_cost_usd:.5f} | "
            f"avg_cost_per_clause={self.avg_cost_per_clause:.5f} | "
            f"success_rate={self.success_rate_percent:.1f}% | "
            f"errors={self.failed_clauses}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "total_clauses": self.total_clauses,
            "successful_clauses": self.successful_clauses,
            "failed_clauses": self.failed_clauses,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_clause": self.avg_cost_per_clause,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "success_rate_percent": self.success_rate_percent,
            "total_execution_time_ms": self.total_execution_time_ms,
        }


class CostTracker:
    """Comprehensive cost tracking and logging for agent operations."""

    def __init__(self, logger_name: str = "veridict.cost_tracker") -> None:
        """Initialize the cost tracker with a named logger."""
        self.logger = logging.getLogger(logger_name)
        self._execution_logs: list[AgentExecutionLog] = []

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost in USD for a given model and token usage.

        Args:
            model: Model name (e.g., "gpt-5-nano").
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.

        Returns:
            Cost in USD.
        """
        pricing = ModelPricing.get_pricing(model)
        input_cost = (input_tokens * pricing["input"]) / 1_000_000
        output_cost = (output_tokens * pricing["output"]) / 1_000_000
        return input_cost + output_cost

    def create_log(
        self,
        agent_type: str,
        clause_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        execution_time_ms: int,
        status: str | ExecutionStatus,
        extra_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
    ) -> AgentExecutionLog:
        """Create an agent execution log entry.

        Args:
            agent_type: Type of agent (e.g., "OBLIGATION_EXTRACTION").
            clause_id: ID of the clause being processed.
            model: Model name used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            execution_time_ms: Execution time in milliseconds.
            status: Execution status.
            extra_data: Additional data to log.
            error_message: Error message if failed.
            retry_count: Number of retries attempted.

        Returns:
            AgentExecutionLog instance.
        """
        if isinstance(status, str):
            status = ExecutionStatus(status)

        cost = self.calculate_cost(model, input_tokens, output_tokens)

        return AgentExecutionLog(
            timestamp=datetime.now(timezone.utc),
            agent_type=agent_type,
            clause_id=clause_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            execution_time_ms=execution_time_ms,
            cost_usd=cost,
            status=status,
            extra_data=extra_data or {},
            error_message=error_message,
            retry_count=retry_count,
        )

    def log_execution(self, log: AgentExecutionLog) -> None:
        """Log an agent execution and store for batch summary.

        Args:
            log: The execution log entry.
        """
        self._execution_logs.append(log)
        log_level = logging.INFO if log.status == ExecutionStatus.SUCCESS else logging.WARNING
        self.logger.log(log_level, log.to_log_string())

    def get_batch_summary(self, batch_id: str) -> BatchCostSummary:
        """Generate a batch summary from accumulated execution logs.

        Args:
            batch_id: Identifier for this batch.

        Returns:
            BatchCostSummary with aggregated metrics.
        """
        if not self._execution_logs:
            return BatchCostSummary(
                batch_id=batch_id,
                total_clauses=0,
                successful_clauses=0,
                failed_clauses=0,
                total_cost_usd=0.0,
                avg_cost_per_clause=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                avg_execution_time_ms=0.0,
                success_rate_percent=0.0,
                total_execution_time_ms=0,
            )

        successful = [log for log in self._execution_logs if log.status == ExecutionStatus.SUCCESS]
        failed = [log for log in self._execution_logs if log.status != ExecutionStatus.SUCCESS]

        total_cost = sum(log.cost_usd for log in self._execution_logs)
        total_input = sum(log.input_tokens for log in self._execution_logs)
        total_output = sum(log.output_tokens for log in self._execution_logs)
        total_time = sum(log.execution_time_ms for log in self._execution_logs)
        total_clauses = len(self._execution_logs)

        return BatchCostSummary(
            batch_id=batch_id,
            total_clauses=total_clauses,
            successful_clauses=len(successful),
            failed_clauses=len(failed),
            total_cost_usd=total_cost,
            avg_cost_per_clause=total_cost / total_clauses if total_clauses > 0 else 0.0,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            avg_execution_time_ms=total_time / total_clauses if total_clauses > 0 else 0.0,
            success_rate_percent=(len(successful) / total_clauses * 100) if total_clauses > 0 else 0.0,
            total_execution_time_ms=total_time,
        )

    def log_batch_summary(self, batch_id: str) -> BatchCostSummary:
        """Generate and log the batch summary.

        Args:
            batch_id: Identifier for this batch.

        Returns:
            BatchCostSummary with aggregated metrics.
        """
        summary = self.get_batch_summary(batch_id)
        self.logger.info(summary.to_log_string())
        return summary

    def reset(self) -> None:
        """Clear all accumulated execution logs."""
        self._execution_logs = []

    def get_all_logs(self) -> list[AgentExecutionLog]:
        """Get all accumulated execution logs.

        Returns:
            List of all execution log entries.
        """
        return list(self._execution_logs)

    def calculate_projected_cost(
        self,
        model: str,
        estimated_clauses: int,
        avg_input_tokens: int = 300,
        avg_output_tokens: int = 200,
    ) -> dict[str, float]:
        """Calculate projected cost for processing a number of clauses.

        Args:
            model: Model to use.
            estimated_clauses: Number of clauses to process.
            avg_input_tokens: Average input tokens per clause.
            avg_output_tokens: Average output tokens per clause.

        Returns:
            Dictionary with cost projections.
        """
        cost_per_clause = self.calculate_cost(model, avg_input_tokens, avg_output_tokens)
        total_cost = cost_per_clause * estimated_clauses

        return {
            "model": model,
            "estimated_clauses": estimated_clauses,
            "cost_per_clause_usd": cost_per_clause,
            "total_projected_cost_usd": total_cost,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
        }


# Global cost tracker instance
cost_tracker = CostTracker()
