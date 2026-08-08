"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

import numpy as np
import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import store


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Redirect every store path into tmp_path so tests never touch the real data/index/."""
    d = tmp_path / "index"
    monkeypatch.setattr(store, "INDEX_DIR", d)
    monkeypatch.setattr(store, "FAISS_PATH", d / "faiss.index")
    monkeypatch.setattr(store, "CHUNKS_PATH", d / "chunks.jsonl")
    monkeypatch.setattr(store, "META_PATH", d / "index_meta.json")
    return d


CFG = {
    "index": {"type": "faiss:flat", "chunk_tokens": 512, "overlap": 0},
    "embed": {"model": "BAAI/bge-m3", "dim": 8},
}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _corpus(n=4, dim=8, seed=0):
    """n chunks with distinct unit vectors, ids/doc_ids/page_ids shaped like the real ones."""
    rng = np.random.default_rng(seed)
    chunks, vecs = [], []
    pages = ["as_p0255", "as_p0255", "as_p0360", "as_p0243"]
    chapters = ["ch06_gamma", "ch06_gamma", "ch09_bessel", "ch05_expint"]
    formulas = ["6.1.8", "6.1.9", "9.1.12", ""]
    for i in range(n):
        page, chapter, fid = pages[i % 4], chapters[i % 4], formulas[i % 4]
        cid = f"{chapter}|{page}|r{i:02d}" + (f"|{fid}" if fid else "")
        chunks.append(
            Chunk(id=cid, doc_id=chapter, text=f"formula body {i}", page_ids=[page], score=0.0)
        )
        vecs.append(_unit(rng.normal(size=dim)))
    return chunks, np.stack(vecs).astype(np.float32)


class TestBuild:
    def test_writes_all_three_artifacts(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "chunks.jsonl").exists()
        assert (index_dir / "index_meta.json").exists()

    def test_meta_records_the_stats_the_form_needs(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        assert meta["n_chunks"] == 4
        assert meta["embedding_dim"] == 8
        assert meta["index_type"] == "faiss:flat"
        assert meta["pages_covered"] == 3  # 0255 twice, 0360, 0243
        assert meta["chapters_covered"] == 3
        assert meta["index_size_bytes"] > 0

    def test_rejects_vector_count_mismatch(self, index_dir):
        chunks, vectors = _corpus(4)
        with pytest.raises(ValueError, match="one row per chunk"):
            store.build(chunks, vectors[:3], CFG)

    def test_rejects_non_2d_vectors(self, index_dir):
        chunks, _ = _corpus(4)
        with pytest.raises(ValueError, match="2-D"):
            store.build(chunks, np.zeros(4, dtype=np.float32), CFG)

    def test_rejects_unknown_index_type(self, index_dir):
        chunks, vectors = _corpus()
        with pytest.raises(ValueError, match="unsupported index type"):
            store.build(chunks, vectors, {"index": {"type": "annoy:hnsw"}, "embed": {}})

    def test_unnormalised_vectors_are_normalised(self, index_dir):
        """Otherwise 'cosine' silently becomes a dot product that ranks by magnitude."""
        chunks, vectors = _corpus()
        loud = vectors * np.array([[1.0], [50.0], [1.0], [1.0]], dtype=np.float32)
        store.build(chunks, loud, CFG)
        loaded = store.load(CFG)
        # query with chunk 0's own vector: it must win, not the 50x-longer chunk 1
        scores, ids = loaded.index.search(vectors[:1], 4)
        assert loaded.chunks[ids[0][0]].id == chunks[0].id
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_empty_corpus_does_not_crash(self, index_dir):
        store.build([], np.zeros((0, 8), dtype=np.float32), CFG)
        assert (index_dir / "index_meta.json").exists()
        assert json.loads((index_dir / "index_meta.json").read_text())["n_chunks"] == 0


class TestLoad:
    def test_round_trips_chunks_faithfully(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)

        assert loaded.index.ntotal == len(chunks)
        assert loaded.dim == 8
        assert loaded.index_type == "faiss:flat"
        assert [c.id for c in loaded.chunks] == [c.id for c in chunks]
        assert [c.doc_id for c in loaded.chunks] == [c.doc_id for c in chunks]
        assert [c.text for c in loaded.chunks] == [c.text for c in chunks]
        assert [c.page_ids for c in loaded.chunks] == [c.page_ids for c in chunks]

    def test_row_order_matches_chunk_order(self, index_dir):
        """The one contract A3's retriever depends on: index row i IS chunks[i]."""
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        for i in range(len(chunks)):
            _scores, ids = loaded.index.search(vectors[i : i + 1], 1)
            assert ids[0][0] == i
            assert loaded.chunks[ids[0][0]].id == chunks[i].id

    def test_is_exact_cosine(self, index_dir):
        """Flat + unit vectors => the top score for a chunk's own vector is exactly 1.0."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        scores, _ids = loaded.index.search(vectors[2:3], 1)
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_unpacks_positionally_and_by_name(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        index, got_chunks, dim, index_type = loaded
        assert index is loaded.index and got_chunks == loaded.chunks
        assert dim == loaded.dim and index_type == loaded.index_type

    def test_missing_index_raises_with_the_fix(self, index_dir):
        with pytest.raises(FileNotFoundError, match="build_index.sh"):
            store.load(CFG)

    def test_inconsistent_sidecar_is_caught(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        rows = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        (index_dir / "chunks.jsonl").write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="inconsistent"):
            store.load(CFG)

    def test_citation_metadata_survives_the_round_trip(self, index_dir):
        """A retrieved chunk must still carry its page and formula id, or we cannot cite."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        gamma = next(c for c in loaded.chunks if c.id.endswith("|6.1.8"))
        assert gamma.page_ids == ["as_p0255"]
        assert gamma.doc_id == "ch06_gamma"


class TestRebuild:
    def test_rebuild_replaces_rather_than_appends(self, index_dir):
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        store.build(chunks[:2], vectors[:2], CFG)
        loaded = store.load(CFG)
        assert loaded.index.ntotal == 2
        assert len(loaded.chunks) == 2

    def test_unicode_maths_survives_the_sidecar(self, index_dir):
        chunks, vectors = _corpus(1)
        chunks[0] = Chunk(
            id=chunks[0].id,
            doc_id=chunks[0].doc_id,
            text=r"\Gamma(\tfrac12)=\pi^{1/2} — Γ(½)=√π",
            page_ids=chunks[0].page_ids,
            score=0.0,
        )
        store.build(chunks, vectors, CFG)
        assert store.load(CFG).chunks[0].text == chunks[0].text
