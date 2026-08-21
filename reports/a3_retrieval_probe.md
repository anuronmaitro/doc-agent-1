# A3 Step 4 — recall@k probe: 39 gold TEST pages, dense vs. reranked

**Measured before any `tasks.jsonl` question was written**, so these numbers can't be tuned to
flatter the task suite later (plan_a3.md Step 4). One query per gold page, formed from that
page's own first non-empty text line — a deterministic, no-cherry-picking proxy query, not a
hand-written "ideal" query.

**Source of truth:** real Kaggle GPU run, kernel `anuronmaitro/mathscholar-a3-step04-index-probe`
version 5, status `COMPLETE` (version 4 produced byte-for-byte identical numbers; v5 only fixed
two lint findings — `E402`/`B905` — with no behavior change, confirmed by comparing both runs'
result JSON directly). See the rendered notebook at
[KAGGLE/a3_step04_index_probe/kaggle_step04.ipynb](../KAGGLE/a3_step04_index_probe/kaggle_step04.ipynb)
for the real execution output. Result JSON (`/kaggle/working/a3_retrieval_probe.json`, all 39
per-page ranks) downloaded via `kaggle kernels output` and is reproduced in full below.

## Recall@k

| k  | dense | reranked |
|----|-------|----------|
| 1  | 0.128 (5/39) | 0.179 (7/39) |
| 5  | 0.256 (10/39) | 0.308 (12/39) |
| 10 | 0.308 (12/39) | 0.359 (14/39) |

Reranking (`BAAI/bge-reranker-v2-m3` over the top-40 dense candidates) improves recall at every k
measured. Absolute recall is modest — honestly reported, not adjusted: **17/39 gold pages (44%)
never appeared in the top-40 dense candidates at all** (`dense_rank=None`), so no amount of
reranking those particular queries could have found them; reranking's gain is entirely among
pages the dense stage *did* surface, by promoting them higher.

**Read this number with its query-construction caveat in mind, not as a ceiling on the system's
real retrieval quality**: the query is the gold page's own first text line (often a section
header like `"EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.243)"`), not a realistic user
question. A header-as-query is a weaker retrieval signal than the natural-language questions
`tasks.jsonl` (Step 14) will actually use — this probe is a floor, not the number end-to-end
agent answers will be judged against.

## Per-page detail (all 39, real ranks)

| page_id | dense_rank | reranked_rank | query |
|---|---|---|---|
| as_p0243 | — | — | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.243) |
| as_p0255 | 1 | 1 | 6. Gamma Function and Related Functions. Mathematical Properties. |
| as_p0360 | — | — | BESSEL FUNCTIONS OF INTEGER ORDER (p.360) |
| as_p0229 | 4 | 1 | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.229) |
| as_p0230 | 3 | 1 | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.230) |
| as_p0232 | 1 | 1 | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.232) |
| as_p0234 | — | — | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.234) |
| as_p0242 | — | — | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.242) |
| as_p0247 | 24 | 10 | EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.247) |
| as_p0256 | 36 | 3 | GAMMA FUNCTION AND RELATED FUNCTIONS (p.256) |
| as_p0258 | — | — | GAMMA FUNCTION AND RELATED FUNCTIONS (p.258) |
| as_p0259 | 3 | 15 | GAMMA FUNCTION AND RELATED FUNCTIONS (p.259) |
| as_p0260 | — | — | GAMMA FUNCTION AND RELATED FUNCTIONS (p.260) |
| as_p0262 | — | — | GAMMA FUNCTION AND RELATED FUNCTIONS (p.262) |
| as_p0280 | 27 | 3 | GAMMA FUNCTION AND RELATED FUNCTIONS (p.280) |
| as_p0281 | — | — | GAMMA FUNCTION AND RELATED FUNCTIONS (p.281) |
| as_p0298 | — | — | ERROR FUNCTION AND FRESNEL INTEGRALS (p.298) |
| as_p0301 | 1 | 3 | ERROR FUNCTION AND FRESNEL INTEGRALS (p.301) |
| as_p0302 | — | — | ERROR FUNCTION AND FRESNEL INTEGRALS (p.302) |
| as_p0303 | 1 | 1 | ERROR FUNCTION AND FRESNEL INTEGRALS (p.303) |
| as_p0312 | — | — | ERROR FUNCTION AND FRESNEL INTEGRALS (p.312) |
| as_p0318 | 34 | 1 | ERROR FUNCTION AND FRESNEL INTEGRALS (p.318) |
| as_p0324 | — | — | ERROR FUNCTION AND FRESNEL INTEGRALS (p.324) |
| as_p0361 | — | — | BESSEL FUNCTIONS OF INTEGER ORDER (p.361) |
| as_p0362 | — | — | BESSEL FUNCTIONS OF INTEGER ORDER (p.362) |
| as_p0366 | 6 | 1 | BESSEL FUNCTIONS OF INTEGER ORDER (p.366) |
| as_p0367 | 4 | 14 | BESSEL FUNCTIONS OF INTEGER ORDER (p.367) |
| as_p0368 | 7 | 2 | BESSEL FUNCTIONS OF INTEGER ORDER (p.368) |
| as_p0369 | 19 | 15 | BESSEL FUNCTIONS OF INTEGER ORDER (p.369) |
| as_p0376 | 15 | 3 | BESSEL FUNCTIONS OF INTEGER ORDER (p.376) |
| as_p0379 | 1 | 29 | BESSEL FUNCTIONS OF INTEGER ORDER (p.379) |
| as_p0383 | — | — | BESSEL FUNCTIONS OF INTEGER ORDER (p.383) |
| as_p0385 | 4 | 25 | BESSEL FUNCTIONS OF INTEGER ORDER (p.385) |
| as_p0423 | 27 | 9 | BESSEL FUNCTIONS OF INTEGER ORDER (p.423) |
| as_p0600 | — | — | ELLIPTIC INTEGRALS (p.600) |
| as_p0619 | — | — | ELLIPTIC INTEGRALS (p.619) |
| as_p0621 | — | — | ELLIPTIC INTEGRALS (p.621) |
| as_p0622 | — | — | ELLIPTIC INTEGRALS (p.622) |
| as_p0625 | — | — | ELLIPTIC INTEGRALS (p.625) |

`—` = not found in the top-40 dense candidates (`K_CANDIDATES=40`, matching `cfg.retrieve.k_max`).

## Index measured against

`index: 3543 vectors, 3543 chunks, meta={'n_chunks': 3543, 'embedding_dim': 1024,
'index_type': 'faiss:flat', 'index_size_bytes': 16224037, 'pages_covered': 987,
'chapters_covered': 29, 'embed_model': 'BAAI/bge-m3', 'built_at': '2026-08-14T11:01:35Z'}`
— the real A2 Step 30 index, mounted from the published `mathscholar-index` Kaggle Dataset, not
rebuilt for this probe.

Note: this probe measures **dense + rerank only**, not the Step 3 visual fallback (`is_weak()`
never triggers here since every query goes through the same top-40 dense path regardless of
score) — the visual fallback's own effect is a separate measurement, out of scope for this step.
