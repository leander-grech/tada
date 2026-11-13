"""Unit tests for inference.task_store.TaskStore."""

import time

from inference.task_store import TASK_EXPIRATION_TIME_SECONDS, TaskStatus, TaskStore


class TestCreate:
    def test_returns_task_with_pending_status(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=3)

        assert task.status == TaskStatus.PENDING
        assert task.task_id
        assert task.result is None
        assert task.error is None

    def test_initialises_progress_per_airport(self):
        store = TaskStore()
        task = store.create(airports=["MXP", "BGY"], run_count=5)

        assert set(task.progress.keys()) == {"MXP", "BGY"}
        for prog in task.progress.values():
            assert prog.completed_runs == 0
            assert prog.total_runs == 5


class TestGet:
    def test_returns_none_for_unknown_id(self):
        store = TaskStore()
        assert store.get("nonexistent") is None

    def test_returns_created_task(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)
        assert store.get(task.task_id) is task


class TestUpdateProgress:
    def test_transitions_pending_to_running(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=3)
        assert task.status == TaskStatus.PENDING

        store.update_progress(task.task_id, "MXP", 1)
        assert task.status == TaskStatus.RUNNING

    def test_updates_completed_runs(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=3)

        store.update_progress(task.task_id, "MXP", 2)
        assert task.progress["MXP"].completed_runs == 2

    def test_ignores_unknown_task_id(self):
        store = TaskStore()
        store.update_progress("nonexistent", "MXP", 1)  # should not raise

    def test_ignores_unknown_airport_code(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=3)
        store.update_progress(task.task_id, "UNKNOWN", 1)  # should not raise
        assert task.progress["MXP"].completed_runs == 0


class TestComplete:
    def test_sets_completed_status_and_result(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)

        result = {"AC1": [{"action": 1}]}
        store.complete(task.task_id, result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None

    def test_ignores_unknown_task_id(self):
        store = TaskStore()
        store.complete("nonexistent", {})  # should not raise


class TestFail:
    def test_sets_failed_status_and_error(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)

        store.fail(task.task_id, "something went wrong")

        assert task.status == TaskStatus.FAILED
        assert task.error == "something went wrong"
        assert task.completed_at is not None

    def test_ignores_unknown_task_id(self):
        store = TaskStore()
        store.fail("nonexistent", "error")  # should not raise


class TestTTLCleanup:
    def test_expired_completed_tasks_are_removed(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)
        store.complete(task.task_id, {})

        # Simulate time passing beyond TTL
        task.completed_at = time.monotonic() - TASK_EXPIRATION_TIME_SECONDS - 1

        assert store.get(task.task_id) is None

    def test_non_expired_tasks_are_kept(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)
        store.complete(task.task_id, {})

        # completed_at is set to now, well within TTL
        assert store.get(task.task_id) is not None

    def test_pending_tasks_are_not_cleaned(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)

        # Even if created long ago, pending tasks have no completed_at
        assert store.get(task.task_id) is not None


class TestCancelAndDelete:
    def test_removes_task_and_sets_event(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=3)
        cancel_event = task.cancel_event

        assert store.cancel_and_delete(task.task_id) is True
        assert store.get(task.task_id) is None
        assert cancel_event.is_set()

    def test_returns_false_for_unknown_task(self):
        store = TaskStore()
        assert store.cancel_and_delete("nonexistent") is False

    def test_works_for_completed_task(self):
        store = TaskStore()
        task = store.create(airports=["MXP"], run_count=1)
        store.complete(task.task_id, {"AC1": [{"action": 1}]})

        assert store.cancel_and_delete(task.task_id) is True
        assert store.get(task.task_id) is None
