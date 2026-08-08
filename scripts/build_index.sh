#!/usr/bin/env bash
# =============================================================================
# MathScholar (team 1) — build the vector index.
#
# Runs the whole knowledge-base chain via the Makefile:
#
#   data/pages/  --ingest.loader-->  list[Page]
#                --ingest.preprocess-->  cleaned pages in data/interim/
#                --vision.layout-->  list[Region]   (reading order)
#                --vision.ocr-->  list[Chunk]       (cached in data/ocr/*.mmd)
#                --index.chunk-->  semantic chunks  (bonus E4)
#                --index.embed-->  vectors          (cached in data/index/embed_cache.npz)
#                --index.store-->  data/index/faiss.index + chunks.jsonl + index_meta.json
#
# Prerequisite: `bash scripts/get_data.sh` (fills data/raw/ + data/pages/).
#
# Every stage caches, so a re-run is cheap and an interrupted run resumes:
#   data/interim/*.png            preprocess   (skips pages already current)
#   data/ocr/<page_id>.mmd        OCR          (skips pages already transcribed)
#   data/index/embed_cache.npz    embeddings   (skips chunks whose text is unchanged)
#
# ⚠️ The OCR stage downloads facebook/nougat-base (~1.4 GB) and runs it over every page.
# That is the Step 16 GPU job — on CPU it is an overnight run, not a coffee break.
# The index statistics this prints at the end are what form Section 5 quotes.
#
# Usage:
#   bash scripts/build_index.sh          # full corpus
#   make ingest index                    # identical, this script is the documented entry point
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d data/pages ] || [ -z "$(ls -A data/pages 2>/dev/null)" ]; then
  echo "ERROR: data/pages/ is empty. Run this first:" >&2
  echo "         bash scripts/get_data.sh" >&2
  exit 1
fi

echo "[build_index] pages available: $(ls data/pages/*.png 2>/dev/null | wc -l)"
echo "[build_index] running: make ingest index"

make ingest index

# Surface the statistics on stdout at the end of the run, so they are visible without
# digging back through the OCR logs. index/store.py wrote them here during `make index`.
if [ -f data/index/index_meta.json ]; then
  echo
  echo "[build_index] index statistics (these are the numbers form Section 5 quotes):"
  cat data/index/index_meta.json
else
  echo "ERROR: data/index/index_meta.json was not written — the chain did not reach" >&2
  echo "       index.store.build(). Check the stage that failed above." >&2
  exit 1
fi

echo
echo "[build_index] done. The index is gitignored and rebuilt by this script; never commit it."
