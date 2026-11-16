# retrieval-eval-ab-testing

An evaluation pipeline for the retrieval step of a RAG system. You change the retrieval; it tells you if the change
helps, hurts, or can't be told apart from luck. In CI, it turns a pull request red when retrieval gets worse.

The retrieval itself is basic on purpose: standard, untuned parts (BM25, Titan embeddings, Cohere reranker, HyDE)
that can be switched on and off. They are the object under test. The work is in the measurement.

It runs on the LegalBench-RAG mini set: 776 questions over 72 legal documents, each with the exact answer text.

## Design decisions

- **Every setup gets the same amount of text.** Each setup returns one page of results (5,000 characters). The
  score is the share of the expert-marked answer that lands on that page. With a fixed page size, a setup can't score
  higher just by returning bigger pieces of text.
- **Error bars that account for shared documents.** Many questions ask about the same document, so they tend to
  succeed or fail together. To measure how much of a result is luck, the tool redraws whole documents at random,
  10,000 times (a bootstrap), and watches how much the score moves. Redrawing single questions instead would make
  the results look up to 2 times more certain than they are.
- **The questions were fixed before the answers.** Which setups to compare, and how to judge them, was written down
  before each run, so the comparisons weren't picked after seeing the scores. The rules are in
  [RESULTS.md](RESULTS.md#pre-registration).
- **One change at a time.** Each setup is a list of on/off switches in `setups.toml`. Two setups being compared must
  differ by exactly one switch, so any difference in score comes from that switch. The code refuses to run a
  comparison that breaks this rule.
- **More comparisons, stricter bar.** The more comparisons you make, the more likely one looks good by chance. A
  standard correction (Holm) raises the bar to account for that.
- **"Can't tell" is an allowed answer.** When the error bar includes zero, the report says the result is
  inconclusive instead of naming a winner.
- **The CI check only blocks what it can prove.** On every pull request, GitHub turns the check red only when the
  change is clearly worse, even allowing for luck. With 72 documents it can see drops of a few points, not of 1 point.
- **GitHub never needs AWS access.** The models run on AWS, so scores are computed locally and committed with the
  code. Each results file records which version of the code produced it; if the code has changed since, the check
  goes red until the scores are recomputed.
- **Anyone can re-run it, for free.** The dataset is checked against a fixed hash, the 776 questions are frozen in a
  file, and every AI model answer is saved locally. A re-run costs nothing; recomputing every saved answer from
  scratch costs about $5.

## Proof that it works

- **It catches a silent bug.** [PR #1](https://github.com/maaninou01/retrieval-eval-ab-testing/pull/1) computes the
  reranker's order but never applies it. Nothing crashes, and the gate goes red: -5.3 points [95% CI -7.6, -3.1].
- **It gives clear verdicts, including negative and inconclusive ones.** The reranker helps (+5.3 points). Adding
  BM25 hurts (-1.8 on dense-rerank). HyDE is inconclusive on 72 documents. All tables are in
  [RESULTS.md](RESULTS.md).

## Limits of the evaluation, and next steps

- **Too few documents to see small changes.** With 72 documents, luck alone can move the score by a few points, so
  only bigger gains or losses show up clearly. Seeing a 1-point change would take hundreds to thousands of
  documents. Next: run on the full benchmark (714 documents).
- **GitHub checks the scores but doesn't produce them.** The scores are computed on my machine, because the models
  run on AWS. Each results file records which version of the code it came from, so a forgotten re-run is caught,
  but a hand-edited score would not be. Next: let GitHub run the evaluation itself, with AWS access limited to the
  model calls.
- **The CI check protects only the recommended setup. (1 of 8)** A change that makes another setup worse still passes.
  Next: check every setup on each pull request.
- **It checks the search, not the answer.** It measures whether the right passage was found, not whether the
  answer written from it is correct. Next: grade the final answers on a small sample checked by hand.
- **One kind of text.** All the documents are English legal texts, so the verdicts may not hold elsewhere. Next:
  repeat on a second field, such as medical or technical documents.

## Reproduce

```
uv sync
uv run python data.py                      # download, check the SHA-256, rebuild the mini set
uv run python evaluate.py run [setup ...]  # every setup when none is named; calls AWS unless cache/ holds the answers
uv run python evaluate.py report           # the tables in RESULTS.md
uv run python evaluate.py gate             # what CI runs; needs no AWS
uv run pytest -q
```
