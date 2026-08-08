"""Unit test home for data validation. IMPLEMENT — CI runs these."""

import json

import pytest
import yaml

from doc_agent.contracts import Page
from doc_agent.data.validate import BUILD_CHAPTERS, TEST_CHAPTERS, VAL_CHAPTERS, validate
from doc_agent.data.versioning import snapshot


def _task_yaml(tmp_path, min_pages=2, min_words=5):
    p = tmp_path / "task.yaml"
    p.write_text(
        yaml.safe_dump({"corpus": {"min_pages": min_pages, "min_words": min_words}}),
        encoding="utf-8",
    )
    return p


def _write_ocr(ocr_dir, page_id, text):
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / f"{page_id}.mmd").write_text(text, encoding="utf-8")


def _pages(n=2, chapter="ch06_gamma"):
    return [
        Page(id=f"as_p{i:04d}", image_path=f"data/pages/as_p{i:04d}.png", doc_id=chapter)
        for i in range(1, n + 1)
    ]


class TestValidate:
    def test_passes_with_enough_pages_and_ocr_words(self, tmp_path):
        pages = _pages(2)
        ocr_dir = tmp_path / "ocr"
        _write_ocr(ocr_dir, "as_p0001", "one two three")
        _write_ocr(ocr_dir, "as_p0002", "four five")
        task = _task_yaml(tmp_path, min_pages=2, min_words=5)
        validate(pages, ocr_dir=ocr_dir, task_path=task)  # must not raise

    def test_rejects_too_few_pages(self, tmp_path):
        pages = _pages(1)
        ocr_dir = tmp_path / "ocr"
        _write_ocr(ocr_dir, "as_p0001", "one two three four five")
        task = _task_yaml(tmp_path, min_pages=2, min_words=1)
        with pytest.raises(ValueError, match="pages"):
            validate(pages, ocr_dir=ocr_dir, task_path=task)

    def test_rejects_insufficient_ocr_words(self, tmp_path):
        pages = _pages(2)
        ocr_dir = tmp_path / "ocr"
        _write_ocr(ocr_dir, "as_p0001", "one")
        # as_p0002 has no OCR file at all — must not silently count as 0 words and pass
        task = _task_yaml(tmp_path, min_pages=2, min_words=100)
        with pytest.raises(ValueError, match="words"):
            validate(pages, ocr_dir=ocr_dir, task_path=task)

    def test_archive_text_layer_cannot_satisfy_the_word_floor(self, tmp_path):
        """Word count must come from OUR OCR — an empty ocr_dir must fail, not fall back."""
        pages = _pages(2)
        ocr_dir = tmp_path / "ocr_never_written"
        task = _task_yaml(tmp_path, min_pages=2, min_words=1)
        with pytest.raises(ValueError, match="words"):
            validate(pages, ocr_dir=ocr_dir, task_path=task)

    def test_rejects_unknown_chapter(self, tmp_path):
        pages = _pages(2, chapter="ch99_not_a_real_chapter")
        ocr_dir = tmp_path / "ocr"
        _write_ocr(ocr_dir, "as_p0001", "one two three four five")
        _write_ocr(ocr_dir, "as_p0002", "six seven")
        task = _task_yaml(tmp_path, min_pages=2, min_words=1)
        with pytest.raises(ValueError, match="chapter"):
            validate(pages, ocr_dir=ocr_dir, task_path=task)

    def test_rejects_conflicting_doc_id_for_same_page_id(self, tmp_path):
        pages = [
            Page(id="as_p0001", image_path="x.png", doc_id="ch06_gamma"),
            Page(id="as_p0001", image_path="x.png", doc_id="ch09_bessel"),
        ]
        ocr_dir = tmp_path / "ocr"
        _write_ocr(ocr_dir, "as_p0001", "one two three")
        task = _task_yaml(tmp_path, min_pages=1, min_words=1)
        with pytest.raises(ValueError, match="doc_id"):
            validate(pages, ocr_dir=ocr_dir, task_path=task)

    def test_rejects_empty_page_list(self):
        with pytest.raises(ValueError):
            validate([])

    def test_rejects_non_page_element(self, tmp_path):
        task = _task_yaml(tmp_path, min_pages=1, min_words=1)
        with pytest.raises(TypeError):
            validate(["not a page"], ocr_dir=tmp_path / "ocr", task_path=task)  # type: ignore[list-item]

    def test_build_val_test_chapters_are_pairwise_disjoint_and_cover_29(self):
        assert BUILD_CHAPTERS.isdisjoint(VAL_CHAPTERS)
        assert BUILD_CHAPTERS.isdisjoint(TEST_CHAPTERS)
        assert VAL_CHAPTERS.isdisjoint(TEST_CHAPTERS)
        assert len(BUILD_CHAPTERS) == 20
        assert len(VAL_CHAPTERS) == 4
        assert len(TEST_CHAPTERS) == 5

    def test_a1_gold_pages_land_in_test_chapters(self):
        # printed 243, 255, 360 -> ch05_expint, ch06_gamma, ch09_bessel (data/provenance.md)
        assert {"ch05_expint", "ch06_gamma", "ch09_bessel"} <= TEST_CHAPTERS


class TestSnapshot:
    def test_returns_stable_12_char_id_for_same_contents(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "as_p0001.png").write_bytes(b"a" * 10)
        (corpus / "as_p0002.png").write_bytes(b"b" * 20)

        out = tmp_path / "v1.json"
        v1 = snapshot(str(corpus), out_path=out)
        v2 = snapshot(str(corpus), out_path=out)

        assert v1 == v2
        assert len(v1) == 12

    def test_id_changes_when_a_file_changes(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "as_p0001.png").write_bytes(b"a" * 10)
        v1 = snapshot(str(corpus), out_path=tmp_path / "v.json")

        (corpus / "as_p0001.png").write_bytes(b"a" * 11)  # size changed
        v2 = snapshot(str(corpus), out_path=tmp_path / "v.json")

        assert v1 != v2

    def test_writes_a_reproducible_record(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "as_p0001.png").write_bytes(b"a" * 10)
        out = tmp_path / "corpus_version.json"

        version_id = snapshot(str(corpus), out_path=out)

        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["version_id"] == version_id
        assert record["n_files"] == 1
        assert record["total_bytes"] == 10

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            snapshot(str(tmp_path / "does_not_exist"))

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            snapshot(str(empty))
