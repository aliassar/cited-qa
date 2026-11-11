"""
Retrieval: pick the passages that should be shown to the model.

WHY RETRIEVAL AT ALL (the "R" in RAG)
--------------------------------------
You cannot paste a 4,000-page matter into a prompt: context windows are finite,
tokens cost money, latency scales with input, and - counter-intuitively -
answers get WORSE as you add irrelevant text, because the signal is diluted.
So: find the handful of passages that actually bear on the question, and send
only those. Grounding the model in real text is also what makes citations
possible, and citations are the point of this API.

WHAT THIS IMPLEMENTS: BM25
--------------------------
BM25 is the workhorse lexical ranking function - the default similarity in
Lucene, and therefore in Elasticsearch and OpenSearch. (Postgres full-text
search is NOT BM25: its `ts_rank` / `ts_rank_cd` use their own frequency- and
proximity-based ranking with no IDF term.) Score of a document D for query Q:

    score(D,Q) = SUM over query terms q of
                 IDF(q) * ( f(q,D) * (k1+1) )
                          / ( f(q,D) + k1 * (1 - b + b * |D|/avgdl) )

Read it as three ideas:
  * IDF - rare words are informative. "indemnification" tells you far more
    about which clause is relevant than "the".
  * SATURATION (the k1 term) - the 10th occurrence of a word adds less than the
    2nd. Plain term-frequency counting is fooled by repetition.
  * LENGTH NORMALISATION (the b term) - long passages contain more words by
    accident, so they must be penalised or they always win.

WHY NOT VECTORS HERE
--------------------
Semantic (embedding) search catches paraphrase - "can we end the contract
early?" matching a "Termination for Convenience" clause that shares no words.
Lexical search catches exact terms - defined terms, party names, section
numbers - which is exactly what legal work is full of, and where embeddings are
weakest.

The production answer is HYBRID: run both, then fuse with Reciprocal Rank
Fusion (score = SUM 1/(k + rank_i), k~60), which needs no score calibration
between the two systems. Add a cross-encoder reranker over the top ~50 if
quality still matters more than latency.

We implement the lexical half honestly and in pure Python: no external service,
fully testable, and it makes the mechanics visible. Swapping in pgvector later
is a change to this one module.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Split on non-word characters and lowercase. A real pipeline would also stem
# ("terminating" -> "terminat"); we skip that to keep the scoring maths visible.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A minimal stopword list.
#
# WHY THIS IS NEEDED HERE SPECIFICALLY: IDF is supposed to make "the" worthless,
# and on a large corpus it does - "the" appears everywhere, so its IDF collapses
# to nearly zero. But we build the index over ONE MATTER, which may hold only a
# handful of passages. On a corpus of three passages, a "the" that happens to
# appear in only one of them gets a HIGH idf and can outrank a genuinely
# discriminating term like "terminate". That is a small-corpus artefact of IDF,
# and dropping stopwords is the standard fix.
#
# Trade-off worth knowing: aggressive stopword removal breaks phrase queries
# ("to be or not to be" becomes empty). Search engines therefore keep stopwords
# in the index and handle them at query time instead.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how has have he her his i if in
    is it its me my not of on or our shall she that the their there these they this
    to was we were what when where which who will with would you your""".split()
)

# k1 controls how fast term frequency saturates; b controls length
# normalisation (0 = off, 1 = full). These are the standard defaults and are
# rarely worth tuning before you have an eval set to tune against.
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    text: str
    score: float


@dataclass
class _Doc:
    """One indexed passage: its identity plus its bag of words."""

    chunk_id: str
    document_id: str
    document_title: str
    ordinal: int
    text: str
    tokens: list[str]
    term_freq: Counter
    length: int


class BM25Index:
    """
    An in-memory BM25 index built per query over one matter's chunks.

    Rebuilding per query is O(total tokens in the matter) and is fine at this
    scale (a matter is tens to low thousands of chunks). It would NOT be fine
    across an entire firm: there, the index has to be persistent and
    incrementally updated - Elasticsearch/OpenSearch, or Postgres full-text
    search (`tsvector` + GIN index) if its ranking is good enough.
    """

    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs
        self._n = len(docs)
        self._avgdl = (sum(d.length for d in docs) / self._n) if self._n else 0.0

        # Document frequency: in how many passages does each term appear?
        self._doc_freq: Counter = Counter()
        for doc in docs:
            # set() so a term repeated 20 times in one passage counts once.
            for term in set(doc.tokens):
                self._doc_freq[term] += 1

    def _idf(self, term: str) -> float:
        """
        Robertson/Sparck-Jones IDF with the +0.5 smoothing terms.

        The outer +1 keeps the value positive: without it, a term appearing in
        more than half the passages gets a NEGATIVE idf, and containing it would
        *hurt* a passage's score, which is nonsense.
        """
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self._docs:
            return []

        scored: list[ScoredChunk] = []
        for doc in self._docs:
            score = 0.0
            for term in query_terms:
                tf = doc.term_freq.get(term, 0)
                if tf == 0:
                    continue
                # Denominator grows with passage length -> long passages need
                # proportionally more hits to score the same.
                denominator = tf + BM25_K1 * (
                    1 - BM25_B + BM25_B * (doc.length / self._avgdl if self._avgdl else 1)
                )
                score += self._idf(term) * (tf * (BM25_K1 + 1)) / denominator

            if score > 0:  # never return passages with zero query overlap
                scored.append(
                    ScoredChunk(
                        chunk_id=doc.chunk_id,
                        document_id=doc.document_id,
                        document_title=doc.document_title,
                        ordinal=doc.ordinal,
                        text=doc.text,
                        score=score,
                    )
                )

        # Sort by score desc; ordinal asc is a deterministic tie-breaker, which
        # matters for reproducible tests and reproducible audit trails.
        scored.sort(key=lambda s: (-s.score, s.ordinal))
        return scored[:top_k]


def build_index(rows: list[tuple[str, str, str, int, str]]) -> BM25Index:
    """
    Build an index from (chunk_id, document_id, document_title, ordinal, text).

    Taking plain tuples rather than ORM objects keeps this module free of any
    database dependency - which is exactly why it is trivial to unit test.
    """
    docs: list[_Doc] = []
    for chunk_id, document_id, document_title, ordinal, text in rows:
        tokens = tokenize(text)
        docs.append(
            _Doc(
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                ordinal=ordinal,
                text=text,
                tokens=tokens,
                term_freq=Counter(tokens),
                length=len(tokens),
            )
        )
    return BM25Index(docs)
