# Results

The evidence behind the [README](README.md). Every number comes from `uv run python evaluate.py report`
or from `results/*.json`.

## Verdicts

Recall difference in points, 95% interval from a document bootstrap. "Helps" and "hurts" hold after Holm.

| round | change | on | recall, points [95% CI] | verdict |
|---|---|---|---|---|
| 2 | add the reranker | dense | +5.3 [+3.1, +7.6] | helps |
| 2 | add BM25 | dense-rerank | -1.8 [-3.1, -0.6] | hurts |
| 2 | add HyDE | dense | +2.6 [-1.4, +6.6] | inconclusive |
| 2 | add HyDE | hybrid-rerank | +1.0 [-0.2, +2.3] | inconclusive |
| 1 | add BM25 | dense | -6.4 [-10.2, -2.8] | hurts |
| 1 | 1,250-character chunks | hybrid | +5.7 [+1.1, +11.2] | helps |
| 1 | add the reranker | hybrid | +9.9 [+6.9, +13.0] | helps |

**Use dense + rerank (`dense-rerank`): the highest point estimate, 35.7% recall@5000. BM25 hurts, with or without
the reranker. HyDE is inconclusive on 72 documents.** dense-rerank was never tested against hybrid-rerank-hyde
(35.0%), so "highest" is a point estimate, not a significant win.

## Pre-registration

Rules 1–7 were written before any data was downloaded or any retrieval was run. Rules 8–12 were written with the
HyDE code, before any HyDE run.

Round 1:

1. Comparisons: hybrid vs dense, hybrid vs hybrid-L, hybrid vs hybrid-rerank. BM25 is a reference only.
2. Headline: character recall at a 5,000-character budget. MRR is secondary.
3. hybrid-L chunk size = 90th percentile of gold snippet length, rounded up to a multiple of 250,
   clamped to [1,000, 2,500]. It is set from the profile, before any retrieval run.
4. 95% paired bootstrap resampling **documents**, 10,000 draws. Holm across the 3 comparisons.
5. Mini set first, with frozen question IDs. If any comparison is inconclusive, **all** comparisons
   are re-run on the full set.
6. The gate goes red when the 95% interval of (PR − baseline) on the recommended setup lies entirely
   below 0. The baseline changes only in a dedicated PR.
7. No expected winner: the reranker and BM25 may lose.

Round 2:

8. Comparisons: the four `round2` pairs in `setups.toml`, each differing in one switch: dense vs dense-hyde,
   hybrid-rerank vs hybrid-rerank-hyde, dense vs dense-rerank, dense-rerank vs hybrid-rerank.
9. HyDE: Claude Haiku 4.5 at temperature 0 writes one passage per question (prompt in `retrieve.py`), and that
   passage alone replaces the question in dense search. BM25 and the reranker keep the real question.
10. Same metric, budget, 500-character chunks and bootstrap as rules 2 and 4; Holm across the four round-2 pairs
    only. Round 1's verdicts stand as published.
11. Rule 5 does not apply: an inconclusive pair is reported as inconclusive on 72 documents, with no full-set run.
12. The recommended setup changes only by the author's decision, in its own PR, weighing any gain against HyDE's
    extra model call per search.

Every round-1 comparison is significant after Holm, so rule 5 (re-run on the full set) was not triggered. The HyDE
prompt was reworded after rule 9 was written and before any scored HyDE run, because the first wording made Haiku
refuse.

## Full tables

Printed by `uv run python evaluate.py report`, pasted unedited.

### Setups (776 questions, 72 documents)

| setup | chunks | recall@5000 [95% CI] | MRR@5000 |
|---|---|---|---|
| bm25 | 22,581 | 14.6% [10.8%, 18.7%] | 0.087 |
| dense | 22,581 | 30.4% [24.9%, 35.8%] | 0.192 |
| hybrid | 22,581 | 24.0% [19.7%, 28.4%] | 0.142 |
| hybrid-L | 9,893 | 29.7% [22.6%, 37.1%] | 0.238 |
| hybrid-rerank | 22,581 | 33.9% [28.7%, 38.9%] | 0.246 |
| dense-rerank | 22,581 | 35.7% [30.6%, 40.4%] | 0.252 |
| dense-hyde | 22,581 | 33.0% [27.8%, 37.9%] | 0.246 |
| hybrid-rerank-hyde | 22,581 | 35.0% [29.8%, 39.7%] | 0.250 |

### Recall@5000 per dataset

| setup | privacy_qa | contractnli | maud | cuad |
|---|---|---|---|---|
| bm25 | 12.3% | 32.9% | 0.7% | 12.3% |
| dense | 42.3% | 36.2% | 11.0% | 32.1% |
| hybrid | 28.1% | 38.4% | 5.5% | 24.1% |
| hybrid-L | 34.7% | 49.9% | 3.7% | 30.6% |
| hybrid-rerank | 42.1% | 38.9% | 15.2% | 39.5% |
| dense-rerank | 44.2% | 40.5% | 16.0% | 41.9% |
| dense-hyde | 47.5% | 28.9% | 17.9% | 37.9% |
| hybrid-rerank-hyde | 43.7% | 38.0% | 15.7% | 42.5% |

### Comparisons, round1 (paired, document bootstrap; Holm within the round)

| comparison | recall difference, points [95% CI] | bootstrap p | Holm p | design effect | rho | documents to detect a 1-point change |
|---|---|---|---|---|---|---|
| hybrid - dense | -6.4 [-10.2, -2.8] | 0.0004 | 0.0008 | 3.2 | 0.22 | 2,037 |
| hybrid-L - hybrid | +5.7 [+1.1, +11.2] | 0.0106 | 0.0106 | 4.0 | 0.30 | 3,803 |
| hybrid-rerank - hybrid | +9.9 [+6.9, +13.0] | < 0.0002 | < 0.0006 | 2.0 | 0.10 | 1,387 |

### Comparisons, round2 (paired, document bootstrap; Holm within the round)

| comparison | recall difference, points [95% CI] | bootstrap p | Holm p | design effect | rho | documents to detect a 1-point change |
|---|---|---|---|---|---|---|
| dense-hyde - dense | +2.6 [-1.4, +6.6] | 0.2092 | 0.2092 | 2.1 | 0.11 | 2,429 |
| hybrid-rerank-hyde - hybrid-rerank | +1.0 [-0.2, +2.3] | 0.0920 | 0.1840 | 1.1 | 0.01 | 234 |
| dense-rerank - dense | +5.3 [+3.1, +7.6] | < 0.0002 | < 0.0008 | 1.3 | 0.03 | 758 |
| hybrid-rerank - dense-rerank | -1.8 [-3.1, -0.6] | 0.0034 | 0.0102 | 1.3 | 0.03 | 232 |

- rho uses the plain mean of 10.8 questions per document; documents hold 1 to 43 questions, and the size-weighted mean (15.9) would give a lower rho.
- Documents needed reuses the spread of these large differences; a real 1-point change moves fewer questions, so read it as an order of magnitude.

## What the evaluation shows about the retrieval

The retrieval is basic on purpose; these are the failure patterns the per-question results expose.

- **Wrong document.** Every answer sits in one document, but the whole corpus is searched with no document filter:
  on dense-rerank the top chunk comes from the wrong document for 251 of 776 questions (32%; 63% on contractnli, 51%
  on maud). MAUD's merger agreements are long (median 344,508 characters, against 9,525 to 27,447 on the other
  datasets) and share their wording, so chunks from other mergers outrank the right one: MAUD recall is 16.0%
  against 40.5% to 44.2% elsewhere.
- **Fixed-size chunks.** Chunks are cut by length, not by clause or section, yet 332 of 776 questions have their
  gold text in several separate places, and longer chunks alone gained 5.7 points on hybrid.
- **Template questions.** MAUD's 194 questions are 17 question types asked of 18 agreements; only the party names
  change, so a question says little beyond its type.

## Cost

Model calls each switch adds, and the cost of rebuilding the cache from nothing at AWS Bedrock list prices
(Titan V2 $0.02 per million tokens, Rerank 3.5 $2 per 1,000 queries, Haiku 4.5 $1 / $5 per million input / output
tokens):

| switch | extra calls per question | full rebuild |
|---|---|---|
| bm25 | none (local) | $0 |
| dense | 1 Titan embedding; plus 1 per chunk, once | ≈$0.04 per chunk size (22,581 chunks at 500 characters) |
| rerank | 1 Cohere call | ≈$1.55 per setup (776 calls) |
| hyde | 1 Haiku call; its passage is embedded in place of the question | ≈$0.55 (776 passages) |

Rebuilding everything (two chunk sizes, three rerank setups, one set of HyDE passages) costs about $5.30.
