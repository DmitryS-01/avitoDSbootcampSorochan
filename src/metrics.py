"""Ranking metrics and submission helpers."""

import numpy as np
import pandas as pd


def build_label_matrix(
    calibration: pd.DataFrame,
    article_ids: np.ndarray,
) -> np.ndarray:
    """Convert ground-truth strings to a binary matrix."""
    id_to_column = {
        int(article_id): column for column, article_id in enumerate(article_ids)
    }
    labels = np.zeros((len(calibration), len(article_ids)), dtype=np.float32)
    for row, value in enumerate(calibration["ground_truth"]):
        for article_id in map(int, str(value).split()):
            labels[row, id_to_column[article_id]] = 1.0
    return labels


def top_indices(scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Return top-k column indices in descending order."""
    k = min(k, scores.shape[1])
    indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(scores, indices, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(indices, order, axis=1)


def map_at_10(scores: np.ndarray, labels: np.ndarray) -> float:
    """Calculate MAP@10."""
    ranking = top_indices(scores, 10)
    relevance = np.take_along_axis(labels, ranking, axis=1)
    precision = np.cumsum(relevance, axis=1) / np.arange(1, 11)
    denominator = np.maximum(np.minimum(labels.sum(axis=1), 10), 1)
    return float(np.mean((precision * relevance).sum(axis=1) / denominator))


def reciprocal_rank_scores(
    scores: np.ndarray,
    depth: int = 100,
    offset: float = 10.0,
) -> np.ndarray:
    """Convert a score matrix to reciprocal-rank points."""
    indices = top_indices(scores, min(depth, scores.shape[1]))
    result = np.zeros_like(scores, dtype=np.float32)
    points = 1.0 / (np.arange(indices.shape[1]) + 1.0 + offset)
    result[np.arange(len(scores))[:, None], indices] = points
    return result


def rank_power_scores(
    scores: np.ndarray,
    power: float,
    depth: int = 10,
) -> np.ndarray:
    """Convert top ranks to normalized power-weighted points."""
    depth = min(depth, scores.shape[1])
    indices = top_indices(scores, depth)
    result = np.zeros_like(scores, dtype=np.float32)
    points = ((depth - np.arange(depth)) / depth) ** power
    result[np.arange(len(scores))[:, None], indices] = points
    return result


def make_answer(
    query_ids: pd.Series,
    scores: np.ndarray,
    article_ids: np.ndarray,
) -> pd.DataFrame:
    """Build an answer.csv-compatible dataframe."""
    predictions = article_ids[top_indices(scores, 10)]
    return pd.DataFrame(
        {
            "query_id": query_ids.to_numpy(),
            "answer": [" ".join(map(str, row)) for row in predictions],
        }
    )


def validate_answer(
    answer: pd.DataFrame,
    test: pd.DataFrame,
    article_ids: np.ndarray,
) -> None:
    """Validate submission columns, IDs and duplicates."""
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Expected columns: query_id, answer")
    if len(answer) != len(test):
        raise ValueError("Row count does not match test.f")
    if not answer["query_id"].equals(test["query_id"]):
        raise ValueError("query_id order does not match test.f")

    valid_ids = set(map(int, article_ids))
    for value in answer["answer"]:
        predictions = list(map(int, str(value).split()))
        if len(predictions) != 10:
            raise ValueError("Each answer must contain 10 IDs")
        if len(predictions) != len(set(predictions)):
            raise ValueError("Duplicate article_id in one answer")
        if not set(predictions).issubset(valid_ids):
            raise ValueError("Unknown article_id in answer")
