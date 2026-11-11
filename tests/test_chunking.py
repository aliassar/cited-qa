"""
Unit tests for chunking and BM25.

These are PURE unit tests - no app, no database, no HTTP. They run in
microseconds and pin down the logic that the integration tests only exercise
indirectly. That is the shape of a healthy suite: a wide base of fast unit
tests for algorithms, a thinner layer of integration tests for wiring.
"""

from __future__ import annotations

from app.services.ingestion import chunk_text
from app.services.retrieval import build_index


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_document_is_a_single_chunk() -> None:
    chunks = chunk_text("A short clause about termination.")
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0


def test_paragraphs_are_packed_up_to_the_target_size() -> None:
    paragraph = "word " * 40  # ~200 chars
    text = "\n\n".join([paragraph] * 10)
    chunks = chunk_text(text, target=500, overlap=50)

    assert len(chunks) > 1
    # No chunk should overshoot the target by more than one whole paragraph -
    # that is what "pack greedily, then flush" guarantees.
    assert all(len(c.text) <= 500 + len(paragraph) for c in chunks)
    # Ordinals must be dense and ordered, because citations reference them.
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_is_split_with_overlap() -> None:
    """A single wall-of-text clause must still be split, and must overlap."""
    text = "x" * 2000
    chunks = chunk_text(text, target=500, overlap=100)

    assert len(chunks) > 1
    # Consecutive chunks must share a region, or a sentence straddling the
    # boundary would be unretrievable from either side.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.char_start < previous.char_end


def test_char_offsets_point_back_into_the_original_text() -> None:
    """Offsets are what let a citation highlight the exact source span."""
    text = "First clause here.\n\nSecond clause here.\n\nThird clause here."
    chunks = chunk_text(text, target=25, overlap=5)
    for chunk in chunks:
        assert 0 <= chunk.char_start <= chunk.char_end <= len(text)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

ROWS = [
    ("c1", "d1", "NDA", 0, "Either party may terminate this agreement with ninety days notice."),
    ("c2", "d1", "NDA", 1, "Confidential information must not be disclosed to third parties."),
    ("c3", "d1", "NDA", 2, "This agreement is governed by the laws of Germany."),
]


def test_bm25_ranks_the_relevant_passage_first() -> None:
    results = build_index(ROWS).search("how do we terminate the agreement", top_k=3)
    assert results[0].chunk_id == "c1"


def test_bm25_returns_nothing_for_a_query_with_no_overlap() -> None:
    """
    No lexical overlap means no result - and that is BM25's real weakness.

    "Can we walk away early?" shares no words with "terminate ... notice", so
    keyword search misses it entirely. This is exactly the gap semantic
    (embedding) retrieval fills, and why production systems run both and fuse
    the rankings.
    """
    assert build_index(ROWS).search("quantum chromodynamics", top_k=3) == []


def test_bm25_respects_top_k() -> None:
    results = build_index(ROWS).search("agreement", top_k=1)
    assert len(results) == 1


def test_empty_index_is_safe() -> None:
    assert build_index([]).search("anything", top_k=5) == []
