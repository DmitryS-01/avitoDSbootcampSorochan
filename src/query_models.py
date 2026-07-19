"""Models trained on labeled calibration queries."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from metrics import reciprocal_rank_scores


def _knn_scores(
    train_char: sparse.csr_matrix,
    valid_char: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    mode: str = "sum",
    neighbour_count: int = 80,
    power: float = 3.0,
) -> np.ndarray:
    """Transfer article labels from similar calibration queries."""
    similarities = (valid_char @ train_char.T).toarray().astype(np.float32)
    neighbour_count = min(neighbour_count, len(labels))
    neighbours = np.argpartition(
        -similarities,
        kth=neighbour_count - 1,
        axis=1,
    )[:, :neighbour_count]
    weights = np.take_along_axis(similarities, neighbours, axis=1)

    if mode == "max":
        result = np.zeros(
            (valid_char.shape[0], labels.shape[1]),
            dtype=np.float32,
        )
        for column in range(neighbour_count):
            contribution = weights[:, column, None] * labels[neighbours[:, column]]
            result = np.maximum(result, contribution)
        return result

    if mode == "softmax":
        weights = np.exp((weights - weights.max(axis=1, keepdims=True)) / 0.07)
    else:
        weights = np.maximum(weights, 0.0) ** power

    weight_matrix = np.zeros_like(similarities, dtype=np.float32)
    weight_matrix[
        np.arange(valid_char.shape[0])[:, None],
        neighbours,
    ] = weights
    return weight_matrix @ labels


def _duplicated_classifier(
    train_matrix: sparse.csr_matrix,
    valid_matrix: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    model_name: str,
    parameter: float,
) -> np.ndarray:
    """Fit a multiclass model after duplicating multilabel queries."""
    rows, targets = np.where(labels > 0)
    duplicated = train_matrix[rows]

    if model_name == "svc":
        model = LinearSVC(C=parameter, random_state=42)
        model.fit(duplicated, targets)
        prediction = model.decision_function(valid_matrix)
        fill_value = -10.0
    elif model_name == "cnb":
        model = ComplementNB(alpha=parameter)
        model.fit(duplicated, targets)
        prediction = model.predict_log_proba(valid_matrix)
        fill_value = -100.0
    else:
        model = MultinomialNB(alpha=parameter)
        model.fit(duplicated, targets)
        prediction = model.predict_log_proba(valid_matrix)
        fill_value = -100.0

    result = np.full(
        (valid_matrix.shape[0], labels.shape[1]),
        fill_value,
        dtype=np.float32,
    )
    result[:, model.classes_.astype(int)] = prediction
    return result


def _profile_scores(
    train_texts: list[str],
    valid_texts: list[str],
    labels: np.ndarray,
    articles: pd.DataFrame,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
) -> np.ndarray:
    """Search over article profiles built from labeled queries and article text."""
    seen_articles = np.flatnonzero(labels.sum(axis=0) > 0)
    documents: list[str] = []
    for article_column in seen_articles:
        examples = " ".join(
            text
            for text, target in zip(train_texts, labels[:, article_column], strict=True)
            if target > 0
        )
        title = articles.iloc[article_column]["title_clean"]
        body = articles.iloc[article_column]["body_clean"][:4_000]
        documents.append(f"{title} {title} {title} {examples} {body}")

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
        max_features=100_000,
    )
    document_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(valid_texts)

    result = np.full(
        (len(valid_texts), labels.shape[1]),
        -1.0,
        dtype=np.float32,
    )
    result[:, seen_articles] = (
        (query_matrix @ document_matrix.T).toarray().astype(np.float32)
    )
    return result


def score_query_sources(
    train_matrix: sparse.csr_matrix,
    train_char: sparse.csr_matrix,
    valid_matrix: sparse.csr_matrix,
    valid_char: sparse.csr_matrix,
    labels: np.ndarray,
    train_texts: list[str],
    valid_texts: list[str],
    articles: pd.DataFrame,
    link_graph: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate all supervised query-level retrieval signals."""
    knn = _knn_scores(train_char, valid_char, labels)
    knn_max = _knn_scores(
        train_char,
        valid_char,
        labels,
        mode="max",
        neighbour_count=40,
    )
    knn_soft = _knn_scores(
        train_char,
        valid_char,
        labels,
        mode="softmax",
        neighbour_count=40,
    )

    prototypes = normalize(sparse.csr_matrix(labels.T) @ train_matrix)
    rocchio = (valid_matrix @ prototypes.T).toarray().astype(np.float32)

    sources = {
        "knn_char": knn,
        "knn_max": knn_max,
        "knn_soft": knn_soft,
        "rocchio": rocchio,
        "svc_03": _duplicated_classifier(
            train_matrix,
            valid_matrix,
            labels,
            model_name="svc",
            parameter=0.3,
        ),
        "svc_01": _duplicated_classifier(
            train_matrix,
            valid_matrix,
            labels,
            model_name="svc",
            parameter=0.1,
        ),
        "cnb_05": _duplicated_classifier(
            train_matrix,
            valid_matrix,
            labels,
            model_name="cnb",
            parameter=0.5,
        ),
        "cnb_10": _duplicated_classifier(
            train_matrix,
            valid_matrix,
            labels,
            model_name="cnb",
            parameter=1.0,
        ),
        "mnb_005": _duplicated_classifier(
            train_matrix,
            valid_matrix,
            labels,
            model_name="mnb",
            parameter=0.05,
        ),
        "profile_word": _profile_scores(
            train_texts,
            valid_texts,
            labels,
            articles,
            analyzer="word",
            ngram_range=(1, 2),
        ),
        "profile_char": _profile_scores(
            train_texts,
            valid_texts,
            labels,
            articles,
            analyzer="char_wb",
            ngram_range=(3, 6),
        ),
    }

    frequency = labels.sum(axis=0).astype(np.float32)
    sources["popularity"] = np.tile(frequency[None, :], (len(valid_texts), 1))

    robust = (
        reciprocal_rank_scores(sources["knn_char"], 30, 5)
        + reciprocal_rank_scores(sources["svc_03"], 30, 5)
        + reciprocal_rank_scores(sources["cnb_10"], 30, 5)
    )
    cooccurrence = labels.T @ labels
    np.fill_diagonal(cooccurrence, 0)
    sources["cooccurrence"] = robust + 0.01 * (robust @ np.log1p(cooccurrence))
    sources["article_links"] = robust + 0.05 * (robust @ link_graph)
    return sources
