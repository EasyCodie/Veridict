"""Worker Registry - Central registry for stateless MAKER workers.

Enables dynamic worker lookup and parallel execution for voting.
"""

from typing import Dict, List, Type, Optional
import logging

from .base_worker import BaseWorker, WorkerContext, WorkerResult


logger = logging.getLogger(__name__)


class WorkerRegistry:
    """Central registry for all worker types.
    
    Workers register themselves here and can be looked up by task type.
    This enables the voting mechanism to spawn multiple independent workers.
    """
    
    _workers: Dict[str, Type[BaseWorker]] = {}
    
    @classmethod
    def register(cls, worker_class: Type[BaseWorker]) -> Type[BaseWorker]:
        """Register a worker class.
        
        Can be used as a decorator:
            @WorkerRegistry.register
            class MyWorker(BaseWorker):
                TASK_TYPE = "my_task"
        """
        task_type = worker_class.TASK_TYPE
        if task_type in cls._workers:
            logger.warning(f"Overwriting existing worker for task type: {task_type}")
        
        cls._workers[task_type] = worker_class
        logger.info(f"Registered worker: {task_type} -> {worker_class.__name__}")
        return worker_class
    
    @classmethod
    def get(cls, task_type: str) -> Optional[BaseWorker]:
        """Get a new worker instance for the given task type.
        
        Returns a fresh instance each time to ensure statelessness.
        """
        worker_class = cls._workers.get(task_type)
        if worker_class is None:
            logger.error(f"No worker registered for task type: {task_type}")
            return None
        
        return worker_class()
    
    @classmethod
    def list_workers(cls) -> List[dict]:
        """List all registered workers with their descriptions."""
        return [
            {
                "task_type": task_type,
                "class_name": worker_class.__name__,
                "description": worker_class.DESCRIPTION,
            }
            for task_type, worker_class in cls._workers.items()
        ]
    
    @classmethod
    def execute(cls, task_type: str, context: WorkerContext) -> WorkerResult:
        """Execute a task using the appropriate worker.
        
        Convenience method that gets a worker and executes it.
        """
        worker = cls.get(task_type)
        if worker is None:
            return WorkerResult(
                success=False,
                data={},
                reasoning=f"No worker found for task type: {task_type}",
            )
        
        return worker.execute(context)


# Singleton instance for convenience
worker_registry = WorkerRegistry()
