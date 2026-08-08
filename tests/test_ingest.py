"""Unit test home for ingest. IMPLEMENT — CI runs these."""

import json

import cv2
import numpy as np
import pytest

from doc_agent.contracts import Page
from doc_agent.ingest import preprocess


def _synthetic_page(w=600, h=800, rotate_deg=0.0, seed=0):
    """A page-like image: light paper, dark horizontal text lines, thin strokes."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 205, dtype=np.uint8)
    for y in range(60, h - 60, 40):  # "text lines"
        img[y : y + 8, 60 : w - 60] = 40
        img[y + 12 : y + 14, 60 : w - 200] = 70  # thin sub/superscript-like stroke
    img = (img.astype(np.float32) + rng.normal(0, 1.5, img.shape)).clip(0, 255).astype(np.uint8)
    if rotate_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Redirect the module's output dir into tmp_path so tests never touch data/interim."""
    interim = tmp_path / "interim"
    monkeypatch.setattr(preprocess, "INTERIM_DIR", interim)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    return pages_dir, interim


def _make_pages(pages_dir, specs):
    pages = []
    for page_id, chapter, angle in specs:
        f = pages_dir / f"{page_id}.png"
        cv2.imwrite(str(f), _synthetic_page(rotate_deg=angle))
        pages.append(Page(id=page_id, image_path=str(f), doc_id=chapter))
    return pages


CFG = {"seed": 42}


class TestSkewEstimation:
    @pytest.mark.parametrize("injected", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    def test_recovers_injected_rotation(self, injected):
        """The estimator must return the angle that undoes the injected rotation."""
        img = _synthetic_page(rotate_deg=injected)
        est = preprocess._estimate_skew_deg(img)
        assert est == pytest.approx(-injected, abs=0.15)

    def test_straight_page_estimates_near_zero(self):
        assert abs(preprocess._estimate_skew_deg(_synthetic_page())) < 0.15


class TestDeskewDeadZone:
    def test_tiny_skew_leaves_pixels_untouched(self):
        """Below the dead zone the image must be returned byte-identical, not resampled."""
        img = _synthetic_page()
        params = preprocess._params(CFG)
        out, angle = preprocess._deskew(img, params)
        assert abs(angle) < params["deskew_min_angle_deg"]
        assert np.array_equal(out, img)

    def test_real_skew_is_corrected(self):
        img = _synthetic_page(rotate_deg=1.2)
        params = preprocess._params(CFG)
        out, angle = preprocess._deskew(img, params)
        assert abs(angle) >= params["deskew_min_angle_deg"]
        assert not np.array_equal(out, img)
        # after correcting, the residual skew should be much smaller
        assert abs(preprocess._estimate_skew_deg(out)) < 0.3


class TestNoBinarisation:
    def test_output_is_multilevel_grayscale(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch01_math_constants", 0.0)])
        out = preprocess.run(pages, CFG)
        img = cv2.imread(out[0].image_path, cv2.IMREAD_UNCHANGED)
        assert img.dtype == np.uint8
        assert img.ndim == 2, "must stay single-channel grayscale"
        assert len(np.unique(img)) > 2, "hard binarisation would leave only 2 levels"

    def test_clahe_does_not_saturate_the_page(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch01_math_constants", 0.0)])
        out = preprocess.run(pages, CFG)
        img = cv2.imread(out[0].image_path, cv2.IMREAD_GRAYSCALE)
        saturated = np.mean((img == 0) | (img == 255))
        assert saturated < 0.02


class TestRunContract:
    def test_returns_pages_repointed_to_interim_preserving_ids(self, workspace):
        pages_dir, interim = workspace
        specs = [("as_p0243", "ch05_expint", 0.0), ("as_p0255", "ch06_gamma", 0.0)]
        pages = _make_pages(pages_dir, specs)
        out = preprocess.run(pages, CFG)

        assert [p.id for p in out] == [p.id for p in pages]
        assert [p.doc_id for p in out] == [p.doc_id for p in pages]
        for p in out:
            assert p.image_path.startswith(str(interim).replace("\\", "/"))
            assert (interim / f"{p.id}.png").exists()

    def test_empty_input_returns_empty(self):
        assert preprocess.run([], CFG) == []

    def test_missing_source_image_raises(self, workspace):
        pages_dir, interim = workspace
        pages = [Page(id="as_p0001", image_path=str(pages_dir / "gone.png"), doc_id="ch06_gamma")]
        with pytest.raises(FileNotFoundError):
            preprocess.run(pages, CFG)


class TestIdempotence:
    def test_rerun_is_byte_identical(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch06_gamma", 0.8)])
        preprocess.run(pages, CFG)
        first = (interim / "as_p0001.png").read_bytes()
        preprocess.run(pages, CFG)
        assert (interim / "as_p0001.png").read_bytes() == first

    def test_rerun_skips_already_current_pages(self, workspace, monkeypatch):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch06_gamma", 0.0)])
        preprocess.run(pages, CFG)

        calls = []
        real = preprocess._process_one
        monkeypatch.setattr(
            preprocess,
            "_process_one",
            lambda g, p: (calls.append(1), real(g, p))[1],
        )
        preprocess.run(pages, CFG)
        assert calls == [], "second run must not reprocess an up-to-date page"

    def test_changed_params_force_reprocessing(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch06_gamma", 0.0)])
        preprocess.run(pages, CFG)
        before = (interim / "as_p0001.png").read_bytes()

        stronger = {"seed": 42, "preprocess": {"clahe_clip_limit": 4.0}}
        preprocess.run(pages, stronger)
        assert (interim / "as_p0001.png").read_bytes() != before

    def test_manifest_records_params_and_seed(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch06_gamma", 0.0)])
        preprocess.run(pages, CFG)
        manifest = json.loads((interim / preprocess.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["params"]["seed"] == 42
        assert manifest["params"]["clahe_clip_limit"] == preprocess.CLAHE_CLIP_LIMIT
        assert manifest["n_pages"] == 1


class TestParams:
    def test_defaults_when_no_preprocess_block(self):
        p = preprocess._params({"seed": 7})
        assert p["deskew"] and p["denoise"] and p["clahe"]
        assert p["clahe_tile_grid"] == preprocess.CLAHE_TILE_GRID
        assert p["seed"] == 7

    def test_config_block_overrides_defaults(self):
        p = preprocess._params({"seed": 1, "preprocess": {"denoise": False, "clahe_clip_limit": 3}})
        assert p["denoise"] is False
        assert p["clahe_clip_limit"] == 3.0

    def test_stages_can_be_disabled(self, workspace):
        pages_dir, interim = workspace
        pages = _make_pages(pages_dir, [("as_p0001", "ch06_gamma", 0.0)])
        cfg = {"seed": 42, "preprocess": {"deskew": False, "denoise": False, "clahe": False}}
        out = preprocess.run(pages, cfg)
        src = cv2.imread(pages[0].image_path, cv2.IMREAD_GRAYSCALE)
        dst = cv2.imread(out[0].image_path, cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(src, dst), "all stages off must be a pass-through"
