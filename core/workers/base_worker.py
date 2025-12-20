"""Base Worker - Abstract class for stateless MAKER workers.

All workers must inherit from BaseWorker and implement the execute() method.
Workers are stateless - they cannot store state between calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

from core.filtering.reliability_filter import reliability_filter, ReliabilityCheck


@dataclass
class WorkerContext:
    """Minimal context for worker execution.
    
    Contains only what the worker needs - nothing more.
    This prevents hallucination from excess context.
    """
    text: str  # Primary text to process
    title: str = ""  # Optional title/header
    metadata: dict = field(default_factory=dict)  # Task-specific metadata
    
    def __post_init__(self):
        # Enforce text length limits to prevent token overflow
        if len(self.text) > 4000:
            self.text = self.text[:4000]


@dataclass 
class WorkerResult:
    """Standard result from worker execution."""
    success: bool
    data: dict  # Task-specific output data
    confidence: float = 0.5
    confidence_label: str = "Medium"
    reasoning: str = ""
    
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    
    # Reliability tracking
    attempts: int = 1
    discards: int = 0


class BaseWorker(ABC):
    """Abstract base class for all stateless MAKER workers.
    
    Workers must:
    1. Be stateless - no instance variables modified after __init__
    2. Accept only WorkerContext - minimal input
    3. Return WorkerResult - standardized output
    4. Use _call_llm() for LLM calls - includes reliability filtering
    """
    
    # Subclasses must define these
    TASK_TYPE: str = "base"  # Unique identifier for this worker type
    DESCRIPTION: str = "Base worker"  # Human-readable description
    
    # Shared settings
    MAX_RETRIES: int = 3
    
    def __init__(self):
        """Initialize worker. Subclasses should not add instance state."""
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, context: WorkerContext) -> WorkerResult:
        """Execute the worker's task.
        
        Args:
            context: Minimal context for the task.
            
        Returns:
            WorkerResult with task output.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this worker type."""
        pass
    
    def build_user_prompt(self, context: WorkerContext) -> str:
        """Build user prompt from context. Override for custom formatting."""
        return f"""Title: {context.title if context.title else "Not specified"}

Text:
{context.text}
"""
    
    def _validate_result(self, data: dict) -> bool:
        """Validate worker output. Override for custom validation."""
        return True


# Type alias for worker classes
WorkerType = type[BaseWorker]
