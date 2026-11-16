import json

import numpy as np
import pytest

import evaluate
from evaluate import budget_fill, char_recall, doc_bootstrap, holm, reciprocal_rank
from retrieve import rrf


def test_budget_fill_cuts_the_last_chunk_to_the_budget_and_drops_the_rest():
    ranked = [("a", 0, 3000), ("b", 100, 2100), ("a", 5000, 6000)]
    assert budget_fill(ranked, 5000) == [("a", 0, 3000), ("b", 100, 2100)]
    assert budget_fill(ranked, 4000) == [("a", 0, 3000), ("b", 100, 1100)]


def test_recall_counts_only_gold_characters_in_the_gold_document():
    spans = [(100, 200), (300, 400)]
    filled = [("a", 150, 350), ("b", 0, 1000), ("a", 390, 395)]
    assert char_recall(filled, "a", spans) == (50 + 50 + 5) / 200


def test_reciprocal_rank_is_one_over_the_first_chunk_touching_gold_and_zero_if_none():
    spans = [(100, 200)]
    assert reciprocal_rank([("b", 100, 200), ("a", 0, 99), ("a", 199, 300)], "a", spans) == 1 / 3
    assert reciprocal_rank([("a", 0, 100), ("a", 200, 300)], "a", spans) == 0.0


def test_bootstrap_of_zero_differences_has_a_zero_width_interval():
    mean, low, high, p, se = doc_bootstrap([0.0] * 6, ["x", "x", "y", "y", "z", "z"])
    assert (mean, low, high, p, se) == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_holm_multiplies_the_sorted_p_values_by_3_2_1_and_keeps_them_increasing():
    assert holm([0.04, 0.01, 0.03]) == [0.06, 0.03, 0.06]


def test_rrf_puts_a_chunk_ranked_first_and_third_above_one_ranked_second_twice():
    assert rrf([np.array([[0, 1, 2]]), np.array([[2, 1, 0]])]).tolist() == [[0, 2, 1]]


def test_rrf_of_one_ranking_returns_it_unchanged():
    ranking = np.array([[2, 0, 3, 1]])
    assert rrf([ranking]).tolist() == ranking.tolist()


def test_setups_reject_no_retriever_hyde_without_dense_and_a_pair_differing_in_two_keys():
    with pytest.raises(SystemExit, match="needs bm25 or dense"):
        evaluate.parse_setups("[setups.a]\nchunk_size = 500\nrerank = true\n[comparisons]\n")
    with pytest.raises(SystemExit, match="without dense"):
        evaluate.parse_setups("[setups.a]\nchunk_size = 500\nbm25 = true\nhyde = true\n[comparisons]\n")
    two = "[setups.a]\nchunk_size = 500\nbm25 = true\n[setups.b]\nchunk_size = 500\ndense = true\n"
    with pytest.raises(SystemExit, match="not exactly one key"):
        evaluate.parse_setups(two + '[comparisons]\nround = [["a", "b"]]\n')


def _gate(tmp_path, monkeypatch, fingerprint: str, drop: float) -> None:
    """Run the gate on 6 synthetic questions over 3 documents, with results `drop` below the baseline."""
    questions = [{"dataset": "d", "index": i, "doc": "xyz"[i // 2], "recall": 0.5} for i in range(6)]
    (tmp_path / "results").mkdir()
    (tmp_path / "baseline.json").write_text(json.dumps({"setup": "s", "questions": questions}))
    lowered = [{**q, "recall": q["recall"] - drop} for q in questions]
    (tmp_path / "results" / "s.json").write_text(json.dumps({"fingerprint": fingerprint, "questions": lowered}))
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "fingerprint", lambda setup: "current")
    evaluate.gate()


def test_gate_passes_identical_results(tmp_path, monkeypatch):
    _gate(tmp_path, monkeypatch, "current", 0.0)


def test_gate_fails_on_a_stale_fingerprint(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="stale"):
        _gate(tmp_path, monkeypatch, "old", 0.0)


def test_gate_fails_when_the_whole_interval_is_below_zero(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="entirely below 0"):
        _gate(tmp_path, monkeypatch, "current", 0.1)
