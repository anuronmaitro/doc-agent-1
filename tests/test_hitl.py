"""Unit test home for agent/hitl.py + agent/hitl_store.py. IMPLEMENT — CI runs these."""

import pytest

from doc_agent.agent import hitl, hitl_store


@pytest.fixture(autouse=True)
def _queue_to_tmp_path(tmp_path, monkeypatch):
    """Every test writes to an isolated queue file, never the real repo's
    data/hitl_queue.json."""
    monkeypatch.setattr(hitl_store, "QUEUE_PATH", tmp_path / "hitl_queue.json")


class TestHitlStore:
    def test_enqueue_returns_a_usable_id_and_pending_reflects_it(self):
        item_id = hitl_store.enqueue({"reason": "low confidence"})
        pending = hitl_store.pending()
        assert len(pending) == 1
        assert pending[0]["id"] == item_id
        assert pending[0]["reason"] == "low confidence"
        assert pending[0]["status"] == "pending"

    def test_ids_are_unique_across_enqueues(self):
        id1 = hitl_store.enqueue({"reason": "a"})
        id2 = hitl_store.enqueue({"reason": "b"})
        assert id1 != id2

    def test_survives_a_fresh_read_simulating_a_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hitl_store, "QUEUE_PATH", tmp_path / "hitl_queue.json")
        hitl_store.enqueue({"reason": "persisted"})
        # A brand new call to pending() re-reads from disk -- no in-memory cache to rely on.
        assert len(hitl_store.pending()) == 1
        assert hitl_store.pending()[0]["reason"] == "persisted"

    def test_pending_on_a_missing_file_is_an_empty_list_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hitl_store, "QUEUE_PATH", tmp_path / "does_not_exist.json")
        assert hitl_store.pending() == []

    def test_a_corrupt_queue_file_degrades_to_empty_rather_than_raising(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "hitl_queue.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        monkeypatch.setattr(hitl_store, "QUEUE_PATH", path)
        assert hitl_store.pending() == []
        # enqueue() must still work afterwards, overwriting the corrupt file cleanly.
        hitl_store.enqueue({"reason": "recovered"})
        assert len(hitl_store.pending()) == 1

    def test_creates_the_data_directory_if_missing(self, tmp_path, monkeypatch):
        nested = tmp_path / "does" / "not" / "exist" / "hitl_queue.json"
        monkeypatch.setattr(hitl_store, "QUEUE_PATH", nested)
        hitl_store.enqueue({"reason": "x"})
        assert nested.parent.is_dir()


class TestHitlEscalate:
    def test_escalate_queues_and_returns_ok_true_with_the_item_id(self):
        result = hitl.escalate("prompt injection detected", {"action": {"tool": "read_page"}})
        assert result.ok is True
        assert result.payload["escalated"] is True
        assert "item_id" in result.payload
        pending = hitl_store.pending()
        assert len(pending) == 1
        assert pending[0]["id"] == result.payload["item_id"]
        assert pending[0]["context"] == {"action": {"tool": "read_page"}}

    def test_escalate_never_raises_even_if_the_store_is_broken(self, monkeypatch):
        def _boom(_item):
            raise OSError("disk full")

        monkeypatch.setattr(hitl_store, "enqueue", _boom)
        result = hitl.escalate("reason", {})
        assert result.ok is False
        assert "escalation failed" in result.payload["reason"]

    def test_review_queue_returns_pending_items(self):
        hitl.escalate("r1", {})
        hitl.escalate("r2", {})
        queue = hitl.review_queue()
        assert len(queue) == 2
        assert {item["reason"] for item in queue} == {"r1", "r2"}
