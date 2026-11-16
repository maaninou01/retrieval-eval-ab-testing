"""Scoring and statistics. CLI: `python evaluate.py run [setup ...]`, `python evaluate.py report` or `python evaluate.py gate`."""

import hashlib
import json
import math
import sys
import tomllib
from collections.abc import Iterable
from statistics import NormalDist

import numpy as np

import data
import retrieve
from data import ROOT, Chunk

# Pre-registered rule 2: the headline metric is character recall at this budget.
BUDGET_CHARS = 5000
# Pre-registered rule 4: 10,000 document-level bootstrap draws, 95% interval.
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
SWITCHES = {"bm25", "dense", "hyde", "rerank"}
# Conventional power-analysis targets: 5% two-sided false positives, 80% power.
ALPHA = 0.05
POWER = 0.8
# The change the Comparisons tables size a detecting test for: 1 point of recall.
DETECT_CHANGE = 0.01
# Everything a setup's scores depend on; a change to any of them makes its results stale.
FINGERPRINTED = ["data.py", "retrieve.py", "evaluate.py", "question_ids.json"]


def parse_setups(text: str) -> tuple[dict[str, dict], dict[str, list[tuple[str, str]]]]:
    """The setups and each round's (base, change) pairs from setups.toml; exit on a setup or pair that breaks its rules."""
    file = tomllib.loads(text)
    setups = file["setups"]
    rounds = {name: [tuple(pair) for pair in pairs] for name, pairs in file["comparisons"].items()}
    for name, setup in setups.items():
        switches = setup.keys() - {"chunk_size"}
        if not switches <= SWITCHES or not all(setup[s] is True for s in switches):
            sys.exit(f"setups: {name} may only set chunk_size and turn on (= true) {sorted(SWITCHES)}")
        if not setup.keys() & {"bm25", "dense"}:
            sys.exit(f"setups: {name} needs bm25 or dense")
        if "hyde" in setup and "dense" not in setup:
            sys.exit(f"setups: {name} turns on hyde, which only changes dense search, without dense")
    for base, change in (pair for pairs in rounds.values() for pair in pairs):
        a, b = setups[base], setups[change]
        differ = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if len(differ) != 1:
            sys.exit(f"setups: {base} vs {change} differ in {sorted(differ)}, not exactly one key")
    return setups, rounds


SETUPS, ROUNDS = parse_setups((ROOT / "setups.toml").read_text(encoding="utf-8"))


def budget_fill(ranked: Iterable[Chunk], budget: int = BUDGET_CHARS) -> list[Chunk]:
    """Chunks in rank order until the budget is spent; the last one is cut to fit."""
    filled = []
    for doc, start, end in ranked:
        if budget <= 0:
            break
        end = min(end, start + budget)
        filled.append((doc, start, end))
        budget -= end - start
    return filled


def _overlap(chunk: Chunk, doc: str, spans: list[tuple[int, int]]) -> int:
    chunk_doc, start, end = chunk
    return sum(max(0, min(end, e) - max(start, s)) for s, e in spans) if chunk_doc == doc else 0


def char_recall(filled: list[Chunk], doc: str, spans: list[tuple[int, int]]) -> float:
    """Gold characters covered by the filled chunks, divided by all gold characters."""
    return sum(_overlap(c, doc, spans) for c in filled) / sum(e - s for s, e in spans)


def reciprocal_rank(filled: list[Chunk], doc: str, spans: list[tuple[int, int]]) -> float:
    """1 / rank of the first filled chunk that touches gold; 0 if none fits in the budget."""
    return next((1 / rank for rank, c in enumerate(filled, 1) if _overlap(c, doc, spans)), 0.0)


def doc_bootstrap(values: list[float], docs: list[str]) -> tuple[float, float, float, float, float]:
    """Mean, 95% interval, two-sided bootstrap p-value and standard error, resampling documents (with all their questions)."""
    names, doc_ids = np.unique(docs, return_inverse=True)
    sums = np.bincount(doc_ids, weights=values, minlength=len(names))
    counts = np.bincount(doc_ids, minlength=len(names))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(names), (BOOTSTRAP_DRAWS, len(names)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return float(np.mean(values)), float(low), float(high), float(p), float(means.std())


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(ps)
    adjusted = np.maximum.accumulate([min(1.0, (len(ps) - i) * ps[j]) for i, j in enumerate(order)])
    out = np.empty(len(ps))
    out[order] = adjusted
    return out.tolist()


def fingerprint(setup: str) -> str:
    """sha256 of the setup's switches and the code and question list that produced its results."""
    switches = json.dumps(SETUPS[setup], sort_keys=True).encode()
    return hashlib.sha256(switches + b"".join((ROOT / p).read_bytes() for p in FINGERPRINTED)).hexdigest()


def run(*names: str) -> None:
    """Score the named setups, or every setup in setups.toml when none is named."""
    questions, corpus = data.load()
    for setup in names or SETUPS:
        chunks, rankings = retrieve.run(SETUPS[setup], questions, corpus)
        scores = []
        for q, ranking in zip(questions, rankings):
            filled = budget_fill(chunks[i] for i in ranking)
            scores.append({"dataset": q["dataset"], "index": q["index"], "doc": q["doc"],
                           "recall": char_recall(filled, q["doc"], q["spans"]),
                           "rr": reciprocal_rank(filled, q["doc"], q["spans"])})
        out = ROOT / "results" / f"{setup}.json"
        out.parent.mkdir(exist_ok=True)
        result = {"setup": setup, "fingerprint": fingerprint(setup), "chunks": len(chunks), "questions": scores}
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        docs = [s["doc"] for s in scores]
        recall, low, high, _, _ = doc_bootstrap([s["recall"] for s in scores], docs)
        print(f"{setup}: {len(chunks)} chunks, {len(scores)} questions")
        print(f"recall@{BUDGET_CHARS} = {recall:.1%} [95% CI {low:.1%}, {high:.1%}], MRR = {np.mean([s['rr'] for s in scores]):.3f}")


def _fmt_p(p: float, zero: bool) -> str:
    """A p-value whose draws never crossed 0 is only known as an upper bound."""
    return f"< {p:.4f}" if zero else f"{p:.4f}"


def report() -> None:
    """Print the RESULTS.md tables from results/."""
    results = {s: json.loads((ROOT / "results" / f"{s}.json").read_text(encoding="utf-8")) for s in SETUPS}
    first = next(iter(results.values()))["questions"]
    keys = [(q["dataset"], q["index"]) for q in first]
    assert all([(q["dataset"], q["index"]) for q in r["questions"]] == keys for r in results.values()), "results are not paired"
    docs = [q["doc"] for q in first]
    datasets = list(dict.fromkeys(d for d, _ in keys))
    recall = {s: np.array([q["recall"] for q in r["questions"]]) for s, r in results.items()}
    n, k = len(docs), len(set(docs))
    z = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)

    print(f"### Setups ({n} questions, {k} documents)\n")
    print(f"| setup | chunks | recall@{BUDGET_CHARS} [95% CI] | MRR@{BUDGET_CHARS} |")
    print("|---|---|---|---|")
    for s, r in results.items():
        mean, low, high, _, _ = doc_bootstrap(recall[s], docs)
        print(f"| {s} | {r['chunks']:,} | {mean:.1%} [{low:.1%}, {high:.1%}] | {np.mean([q['rr'] for q in r['questions']]):.3f} |")

    print(f"\n### Recall@{BUDGET_CHARS} per dataset\n")
    print("| setup | " + " | ".join(datasets) + " |")
    print("|---|" + "---|" * len(datasets))
    for s in SETUPS:
        cells = [recall[s][[d == name for d, _ in keys]].mean() for name in datasets]
        print(f"| {s} | " + " | ".join(f"{c:.1%}" for c in cells) + " |")

    # Design effect: how much the document clustering inflates the variance over treating questions as independent.
    # Documents needed: the paired, clustered standard error shrinks with the square root of the document count.
    for round_name, pairs in ROUNDS.items():
        rows = []
        for base, change in pairs:
            diff = recall[change] - recall[base]
            mean, low, high, p, se = doc_bootstrap(diff, docs)
            deff = (se / (diff.std(ddof=1) / math.sqrt(n))) ** 2
            rows.append((f"{change} - {base}", mean, low, high, p, deff, (deff - 1) / (n / k - 1), math.ceil(k * (z * se / DETECT_CHANGE) ** 2)))
        print(f"\n### Comparisons, {round_name} (paired, document bootstrap; Holm within the round)\n")
        print(f"| comparison | recall difference, points [95% CI] | bootstrap p | Holm p | design effect | rho "
              f"| documents to detect a {DETECT_CHANGE * 100:g}-point change |")
        print("|---|---|---|---|---|---|---|")
        # With 10,000 draws the smallest non-zero two-sided p-value is 2 / 10,000, so a p of 0 stands for that bound.
        bounds = [max(r[4], 2 / BOOTSTRAP_DRAWS) for r in rows]
        for (name, mean, low, high, p, deff, rho, needed), bound, adjusted in zip(rows, bounds, holm(bounds)):
            print(f"| {name} | {mean * 100:+.1f} [{low * 100:+.1f}, {high * 100:+.1f}] | {_fmt_p(bound, p == 0)} | {_fmt_p(adjusted, p == 0)} "
                  f"| {deff:.1f} | {rho:.2f} | {needed:,} |")
    sizes = np.unique(docs, return_counts=True)[1]
    print(f"\n- rho uses the plain mean of {n / k:.1f} questions per document; documents hold {sizes.min()} to {sizes.max()} questions, "
          f"and the size-weighted mean ({(sizes ** 2).sum() / n:.1f}) would give a lower rho.")
    print("- Documents needed reuses the spread of these large differences; a real 1-point change moves fewer "
          "questions, so read it as an order of magnitude.")



def gate() -> None:
    """Pre-registered rule 6: exit 1 if the baseline setup's results are stale or its recall is significantly lower."""
    baseline = json.loads((ROOT / "baseline.json").read_text(encoding="utf-8"))
    setup = baseline["setup"]
    results = json.loads((ROOT / "results" / f"{setup}.json").read_text(encoding="utf-8"))
    if results.get("fingerprint") != fingerprint(setup):
        sys.exit(f"gate: results/{setup}.json is stale; run `python evaluate.py run {setup}` and commit it")
    keys = [(q["dataset"], q["index"]) for q in baseline["questions"]]
    assert [(q["dataset"], q["index"]) for q in results["questions"]] == keys, "results and baseline are not paired"
    diff = [r["recall"] - b["recall"] for r, b in zip(results["questions"], baseline["questions"])]
    mean, low, high, _, _ = doc_bootstrap(diff, [q["doc"] for q in baseline["questions"]])
    line = f"{setup} recall - baseline = {mean * 100:+.1f} points [95% CI {low * 100:+.1f}, {high * 100:+.1f}]"
    if high < 0:
        sys.exit(f"gate: {line}, entirely below 0")
    print(f"gate: {line}, passed")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"run": run, "report": report, "gate": gate}[command](*args)
