"""In-memory task store for background inference tasks."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AirportProgress:
    completed_runs: int = 0
    total_runs: int = 0


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus
    progress: dict[str, AirportProgress] = field(default_factory=dict)
    result: dict[str, list[dict]] | None = None
    raw_actions: dict[str, list[dict]] | None = None
    error: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


# Completed/failed tasks expire after 2 minutes
TASK_EXPIRATION_TIME_SECONDS = 120


class TaskStore:
    """Thread-safe in-memory task store with TTL cleanup."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create(self, airports: list[str], run_count: int) -> TaskRecord:
        task = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            status=TaskStatus.PENDING,
            progress={code: AirportProgress(total_runs=run_count) for code in airports},
            created_at=time.monotonic(),
        )
        with self._lock:
            self._cleanup()
            self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            self._cleanup()
            return self._tasks.get(task_id)

    def update_progress(self, task_id: str, airport_code: str, completed_runs: int) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING
            progress = task.progress.get(airport_code)
            if progress is not None:
                progress.completed_runs = completed_runs

    def complete(
        self,
        task_id: str,
        result: dict[str, list[dict]],
        raw_actions: dict[str, list[dict]] | None = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.raw_actions = raw_actions
            task.completed_at = time.monotonic()

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.monotonic()

    def cancel_and_delete(self, task_id: str) -> bool:
        """Cancel a task's inference and remove it from the store."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
        if task is None:
            return False
        task.cancel_event.set()
        return True

    def _cleanup(self) -> None:
        """Remove expired completed/failed tasks. Must be called under lock."""
        now = time.monotonic()
        expired = [
            tid
            for tid, task in self._tasks.items()
            if task.completed_at is not None
            and (now - task.completed_at) > TASK_EXPIRATION_TIME_SECONDS
        ]
        for tid in expired:
            del self._tasks[tid]
