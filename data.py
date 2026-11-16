"""LegalBench-RAG data: download, the frozen mini set, loading, chunking, profiling."""

import hashlib
import io
import json
import math
import os
import random
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
ZIP = ROOT / "data" / "legalbenchrag.zip"
QUESTION_IDS = ROOT / "question_ids.json"

# A chunk of the corpus: (document, start, end).
Chunk = tuple[str, int, int]

# The authors' Dropbox folder (linked from the zeroentropy-ai/legalbenchrag README); dl=1 serves it as a zip.
DATA_URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
    "?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1"
)
# 90,591,976 bytes; two downloads were byte-identical.
DATA_SHA256 = "27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d"

# Upstream benchmark.py: benchmark_name_to_weight keys and MAX_TESTS_PER_BENCHMARK.
DATASETS = ("privacy_qa", "contractnli", "maud", "cuad")
MINI_PER_DATASET = 194
# The mini set's 776 questions use 72 documents.
MINI_DOCUMENTS = 72


def _verified(blob: bytes) -> bytes:
    digest = hashlib.sha256(blob).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"data zip SHA-256 is {digest}, pinned {DATA_SHA256}")
    return blob


def download() -> None:
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        blob = _verified(response.read())
    ZIP.parent.mkdir(exist_ok=True)
    partial = ZIP.with_suffix(".partial")
    partial.write_bytes(blob)
    os.replace(partial, ZIP)


class _Archive:
    """Files are read from the zip, not extracted: some corpus names contain ':', which Windows forbids."""

    def __init__(self) -> None:
        self.zip = zipfile.ZipFile(io.BytesIO(_verified(ZIP.read_bytes())))
        # The zip stores names in NFC; the benchmark refers to at least one (LECLANCHE + accent) in NFD.
        self.names = {unicodedata.normalize("NFC", name): name for name in self.zip.namelist()}

    def read(self, path: str) -> str:
        with self.zip.open(self.names[unicodedata.normalize("NFC", path)]) as f:
            # Universal newlines, as upstream's open() reads the files, so span offsets line up.
            return io.TextIOWrapper(f, encoding="utf-8").read()

    def tests(self, dataset: str) -> list[dict]:
        return json.loads(self.read(f"benchmarks/{dataset}.json"))["tests"]


def build_mini() -> None:
    """Rebuild upstream's mini set exactly and freeze it as [dataset, index] pairs."""
    archive, ids, docs = _Archive(), [], set()
    for dataset in DATASETS:
        tests = archive.tests(dataset)
        order = range(len(tests))
        if len(tests) > MINI_PER_DATASET:
            # Upstream sorts by a random number seeded with the first snippet's file path, which groups
            # questions by document. Random(seed) seeds exactly like upstream's random.seed(seed).
            order = sorted(order, key=lambda i: random.Random(tests[i]["snippets"][0]["file_path"]).random())
        kept = order[:MINI_PER_DATASET]
        ids += [[dataset, i] for i in kept]
        docs |= {snippet["file_path"] for i in kept for snippet in tests[i]["snippets"]}
    expected = (MINI_PER_DATASET * len(DATASETS), MINI_DOCUMENTS)
    if (len(ids), len(docs)) != expected:
        raise RuntimeError(f"mini set has {len(ids)} questions over {len(docs)} documents, upstream's has {expected}")
    # LF on every platform, so the file hashes the same on Windows and in CI.
    QUESTION_IDS.write_text(json.dumps(ids) + "\n", encoding="utf-8", newline="\n")


def load() -> tuple[list[dict], dict[str, str]]:
    """The frozen questions (dataset, index, query, doc, spans) and the corpus of the documents they use."""
    archive = _Archive()
    tests = {dataset: archive.tests(dataset) for dataset in DATASETS}
    questions, corpus = [], {}
    for dataset, index in json.loads(QUESTION_IDS.read_text(encoding="utf-8")):
        test = tests[dataset][index]
        docs = {snippet["file_path"] for snippet in test["snippets"]}
        assert len(docs) == 1, f"{dataset}[{index}] spans {len(docs)} documents"
        doc = docs.pop()
        if doc not in corpus:
            corpus[doc] = archive.read(f"corpus/{doc}")
        # Each snippet stores its text: matching it proves the offsets line up with how the text was read.
        assert all(corpus[doc][slice(*snippet["span"])] == snippet["answer"] for snippet in test["snippets"]), (
            f"{dataset}[{index}] span text differs from its stored answer"
        )
        spans = sorted(tuple(snippet["span"]) for snippet in test["snippets"])
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:])), f"{dataset}[{index}] spans overlap"
        questions.append({"dataset": dataset, "index": index, "query": test["query"], "doc": doc, "spans": spans})
    return questions, corpus


def chunk(text: str, size: int) -> list[tuple[int, int]]:
    """(start, end) of non-overlapping chunks of at most `size` characters, cut at blank lines, line breaks, spaces, then anywhere."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0, add_start_index=True)
    offsets = [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in splitter.create_documents([text])]
    # Recall adds overlaps chunk by chunk, so chunks must never overlap each other.
    assert all(a[1] <= b[0] for a, b in zip(offsets, offsets[1:])), "splitter returned overlapping chunks"
    return offsets


def profile(questions: list[dict], corpus: dict[str, str]) -> None:
    """Print gold span and document length percentiles and the hybrid-L chunk size (rule 3)."""
    pcts = (10, 50, 90, 99)
    print(f"{'gold span chars':<24}" + "".join(f"p{p:<7}" for p in pcts) + "n")
    for name in (*DATASETS, "all"):
        lengths = [e - s for q in questions if name in (q["dataset"], "all") for s, e in q["spans"]]
        print(f"{name:<24}" + "".join(f"{v:<8.0f}" for v in np.percentile(lengths, pcts)) + str(len(lengths)))
    doc_lengths = [len(text) for text in corpus.values()]
    print(f"{'document chars':<24}" + "".join(f"{v:<8.0f}" for v in np.percentile(doc_lengths, pcts)) + str(len(doc_lengths)))
    p90 = np.percentile([e - s for q in questions for s, e in q["spans"]], 90)
    size = min(max(math.ceil(p90 / 250) * 250, 1000), 2500)
    print(f"rule 3: p90 = {p90:.1f} -> L = {size}")


if __name__ == "__main__":
    if not ZIP.exists():
        download()
    build_mini()
    questions, corpus = load()
    counts = {d: sum(q["dataset"] == d for q in questions) for d in DATASETS}
    print(f"{len(questions)} questions {counts}, {len(corpus)} documents")
    profile(questions, corpus)
