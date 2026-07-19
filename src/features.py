"""Sparse lexical feature spaces for articles and queries."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class SparseBM25:
    """Small sparse BM25 implementation."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 150_000,
    ) -> None:
        """Initialize the count vectorizer."""
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=1,
            max_features=max_features,
        )
        self.matrix: sparse.csr_matrix | None = None

    def fit(self, documents: list[str]) -> Self:
        """Fit vocabulary and BM25 document weights."""
        counts = self.vectorizer.fit_transform(documents)
        counts = counts.tocsr().astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(lengths.mean()), 1.0)
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)

        k1 = 1.5
        b = 0.75
        normalizer = k1 * (1.0 - b + b * lengths / average_length)
        weighted = counts.tocoo(copy=True)
        weighted.data = (
            weighted.data * (k1 + 1.0) / (weighted.data + normalizer[weighted.row])
        )
        self.matrix = weighted.tocsr().multiply(inverse_document_frequency)
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        """Return query-document BM25 scores."""
        if self.matrix is None:
            raise RuntimeError("Call fit before score")
        query_matrix = self.vectorizer.transform(queries)
        return (query_matrix @ self.matrix.T).toarray().astype(np.float32)


@dataclass(frozen=True)
class _TfidfField:
    """One fitted TF-IDF article field."""

    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix


class ArticleFeatureIndex:
    """Fit lexical indexes for separate HTML fields."""

    def __init__(self) -> None:
        """Create empty field indexes."""
        self.bm25 = SparseBM25((1, 2), 120_000)
        self.stem_bm25 = SparseBM25((1, 2), 150_000)
        self.fields: dict[str, _TfidfField] = {}

    @staticmethod
    def _fit_field(
        documents: list[str],
        *,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int = 150_000,
    ) -> _TfidfField:
        """Fit one article field."""
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(documents).tocsr()
        return _TfidfField(vectorizer=vectorizer, matrix=matrix)

    def fit(self, articles: pd.DataFrame) -> Self:
        """Fit all lexical article indexes."""
        titles = articles["title_clean"].tolist()
        bodies = articles["body_clean"].str.slice(0, 30_000).tolist()
        important = articles["important_clean"].tolist()
        anchors = articles["anchor_clean"].tolist()
        alt_text = articles["alt_clean"].tolist()
        intro = articles["intro_clean"].tolist()
        title_stems = articles["title_stem"].tolist()
        body_stems = articles["body_stem"].tolist()

        documents = [
            f"{title} {title} {body}"
            for title, body in zip(titles, bodies, strict=True)
        ]
        stem_documents = [
            f"{title} {title} {body}"
            for title, body in zip(title_stems, body_stems, strict=True)
        ]
        structured = [
            f"{title} {title} {title} {heading} {anchor} {alt} {lead}"
            for title, heading, anchor, alt, lead in zip(
                titles,
                important,
                anchors,
                alt_text,
                intro,
                strict=True,
            )
        ]
        title_heading = [
            f"{title} {title} {heading}"
            for title, heading in zip(titles, important, strict=True)
        ]

        self.bm25.fit(documents)
        self.stem_bm25.fit(stem_documents)
        self.fields = {
            "body_word": self._fit_field(
                bodies,
                ngram_range=(1, 2),
                min_df=2,
                max_features=120_000,
            ),
            "body_char": self._fit_field(
                bodies,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=120_000,
            ),
            "title_char": self._fit_field(
                titles,
                analyzer="char_wb",
                ngram_range=(3, 6),
                max_features=80_000,
            ),
            "stem_tfidf": self._fit_field(
                stem_documents,
                ngram_range=(1, 2),
                min_df=2,
                max_features=150_000,
            ),
            "title_word": self._fit_field(
                titles,
                ngram_range=(1, 3),
            ),
            "title_char_wide": self._fit_field(
                titles,
                analyzer="char_wb",
                ngram_range=(2, 6),
            ),
            "heading_word": self._fit_field(
                important,
                ngram_range=(1, 3),
            ),
            "heading_char": self._fit_field(
                important,
                analyzer="char_wb",
                ngram_range=(3, 6),
            ),
            "anchor_word": self._fit_field(
                anchors,
                ngram_range=(1, 3),
            ),
            "intro_word": self._fit_field(
                intro,
                ngram_range=(1, 3),
            ),
            "intro_char": self._fit_field(
                intro,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
            ),
            "title_heading_word": self._fit_field(
                title_heading,
                ngram_range=(1, 3),
            ),
            "structured_word": self._fit_field(
                structured,
                ngram_range=(1, 3),
            ),
        }
        return self

    def score(self, queries: pd.DataFrame) -> dict[str, np.ndarray]:
        """Score every query against every article field."""
        clean = queries["query_clean"].tolist()
        stems = queries["query_stem"].tolist()
        result = {
            "bm25": self.bm25.score(clean),
            "stem_bm25": self.stem_bm25.score(stems),
        }
        for name, field in self.fields.items():
            values = stems if name == "stem_tfidf" else clean
            query_matrix = field.vectorizer.transform(values)
            result[name] = (query_matrix @ field.matrix.T).toarray().astype(np.float32)
        return result


class QueryFeatureSpace:
    """Word and character TF-IDF space for user questions."""

    def __init__(self) -> None:
        """Initialize vectorizers."""
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=50_000,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=80_000,
        )
        self.word_matrix: sparse.csr_matrix | None = None
        self.char_matrix: sparse.csr_matrix | None = None

    def fit(self, queries: pd.Series) -> Self:
        """Fit vocabularies on calibration queries."""
        values = queries.tolist()
        self.word_matrix = self.word.fit_transform(values).tocsr()
        self.char_matrix = self.char.fit_transform(values).tocsr()
        return self

    def calibration(
        self,
        word_weight: float = 0.2,
    ) -> sparse.csr_matrix:
        """Return combined calibration features."""
        self._check_fitted()
        return _combine_query_features(
            self.word_matrix,
            self.char_matrix,
            word_weight,
        )

    def calibration_char(self) -> sparse.csr_matrix:
        """Return normalized character features."""
        self._check_fitted()
        return normalize(self.char_matrix)

    def transform(
        self,
        queries: pd.Series,
        word_weight: float = 0.2,
    ) -> sparse.csr_matrix:
        """Transform new queries to the combined space."""
        values = queries.tolist()
        return _combine_query_features(
            self.word.transform(values),
            self.char.transform(values),
            word_weight,
        )

    def transform_char(self, queries: pd.Series) -> sparse.csr_matrix:
        """Transform new queries to character features."""
        return normalize(self.char.transform(queries.tolist()))

    def _check_fitted(self) -> None:
        """Ensure calibration matrices exist."""
        if self.word_matrix is None or self.char_matrix is None:
            raise RuntimeError("Call fit before transform")


def _combine_query_features(
    word_matrix: sparse.spmatrix,
    char_matrix: sparse.spmatrix,
    word_weight: float,
) -> sparse.csr_matrix:
    """Combine word and character spaces with cosine normalization."""
    return normalize(
        hstack(
            [
                np.sqrt(word_weight) * word_matrix,
                np.sqrt(1.0 - word_weight) * char_matrix,
            ],
            format="csr",
        )
    )
