"""Unit test home for agent/memory.py. IMPLEMENT — CI runs these."""

from doc_agent.agent.memory import Memory


class TestRecall:
    def test_empty_memory_returns_empty_list(self):
        mem = Memory()
        assert mem.recall("gamma function") == []

    def test_empty_query_returns_empty_list(self):
        mem = Memory()
        mem.add("gamma function recurrence")
        assert mem.recall("") == []
        assert mem.recall("   ") == []

    def test_no_match_returns_empty_list(self):
        mem = Memory()
        mem.add("bessel functions of integer order")
        assert mem.recall("elliptic integrals") == []

    def test_single_match_is_returned(self):
        mem = Memory()
        mem.add("the gamma function satisfies a recurrence relation")
        result = mem.recall("gamma function recurrence")
        assert result == ["the gamma function satisfies a recurrence relation"]

    def test_most_relevant_first(self):
        mem = Memory()
        mem.add("bessel functions of integer order")  # 0 shared words with the query
        mem.add("the gamma function recurrence relation")  # 3 shared words
        mem.add("gamma function")  # 2 shared words, shorter (higher Jaccard)
        result = mem.recall("gamma function recurrence")
        # "gamma function" has fewer distinct words, so its overlap/union ratio is higher
        # even though it shares fewer raw words than the "recurrence relation" entry.
        assert result[0] == "gamma function"
        assert result[1] == "the gamma function recurrence relation"
        assert "bessel functions of integer order" not in result

    def test_ties_broken_by_insertion_order(self):
        mem = Memory()
        mem.add("gamma function value")
        mem.add("gamma function table")
        result = mem.recall("gamma function")
        assert result == ["gamma function value", "gamma function table"]

    def test_scores_arbitrary_item_types_via_str(self):
        """add() takes Any -- a dict/ToolResult-shaped item must still be searchable."""
        mem = Memory()
        mem.add({"tool": "retrieve", "top_chunk": "gamma_recurrence_chunk"})
        mem.add({"tool": "retrieve", "top_chunk": "bessel_order_chunk"})
        result = mem.recall("gamma recurrence")
        assert len(result) == 1
        assert result[0]["top_chunk"] == "gamma_recurrence_chunk"

    def test_is_case_insensitive(self):
        mem = Memory()
        mem.add("GAMMA Function")
        assert mem.recall("gamma function") == ["GAMMA Function"]

    def test_recall_does_not_mutate_stored_items(self):
        mem = Memory()
        item = "gamma function"
        mem.add(item)
        mem.recall("gamma function")
        assert mem.items == [item]

    def test_a_fresh_memory_per_query_starts_clean(self):
        """Memory is working memory for ONE query loop, not a cross-query cache -- a new
        Memory() must not see anything from a previous instance."""
        first = Memory()
        first.add("gamma function recurrence")
        second = Memory()
        assert second.recall("gamma function recurrence") == []
