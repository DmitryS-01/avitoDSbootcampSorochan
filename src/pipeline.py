"""End-to-end training and prediction pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import load_data
from features import ArticleFeatureIndex, QueryFeatureSpace
from metrics import (
    build_label_matrix,
    make_answer,
    validate_answer,
)
from query_models import score_query_sources
from rankers import (
    blend_rankers,
    build_pair_data,
    fit_lgbm_bagged,
    fit_lgbm_stable,
    fit_xgboost_pairwise,
    predict_pairs,
)


def _build_link_graph(articles: pd.DataFrame) -> np.ndarray:
    """Build a normalized graph from links between help articles."""
    article_ids = articles["article_id"].astype(int).to_numpy()
    id_to_column = {
        int(article_id): column for column, article_id in enumerate(article_ids)
    }
    graph = np.zeros((len(articles), len(articles)), dtype=np.float32)
    for row, value in enumerate(articles["internal_links"]):
        for article_id in str(value).split():
            target = id_to_column.get(int(article_id))
            if target is not None and target != row:
                graph[row, target] = 1.0
    return graph / np.maximum(graph.sum(axis=0, keepdims=True), 1.0)


def _build_metadata(
    queries: pd.DataFrame,
    articles: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Create simple pair-level length and token-overlap features."""
    query_sets = [set(text.split()) for text in queries["query_clean"]]
    title_sets = [set(text.split()) for text in articles["title_clean"]]
    body_sets = [set(text.split()) for text in articles["body_clean"]]

    title_overlap = np.zeros((len(queries), len(articles)), dtype=np.float32)
    body_overlap = np.zeros_like(title_overlap)
    for query_row, query_tokens in enumerate(query_sets):
        denominator = max(len(query_tokens), 1)
        for article_column, (title_tokens, body_tokens) in enumerate(
            zip(title_sets, body_sets, strict=True)
        ):
            title_overlap[query_row, article_column] = (
                len(query_tokens & title_tokens) / denominator
            )
            body_overlap[query_row, article_column] = (
                len(query_tokens & body_tokens) / denominator
            )

    return {
        "title_token_recall": title_overlap,
        "body_token_recall": body_overlap,
        "title_length": np.tile(
            np.log1p(articles["title_clean"].str.len().to_numpy())[None, :],
            (len(queries), 1),
        ).astype(np.float32),
        "body_length": np.tile(
            np.log1p(articles["body_length"].to_numpy())[None, :],
            (len(queries), 1),
        ).astype(np.float32),
        "query_length": np.tile(
            np.log1p(queries["query_length"].to_numpy())[:, None],
            (1, len(articles)),
        ).astype(np.float32),
        "query_tokens": np.tile(
            np.log1p(queries["token_count"].to_numpy())[:, None],
            (1, len(articles)),
        ).astype(np.float32),
    }


def _build_oof_query_scores(
    calibration: pd.DataFrame,
    labels: np.ndarray,
    query_space: QueryFeatureSpace,
    articles: pd.DataFrame,
    link_graph: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build out-of-fold supervised features for ranker training."""
    matrix = query_space.calibration()
    char_matrix = query_space.calibration_char()
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    result: dict[str, np.ndarray] = {}

    for train_index, valid_index in splitter.split(calibration):
        fold_scores = score_query_sources(
            matrix[train_index],
            char_matrix[train_index],
            matrix[valid_index],
            char_matrix[valid_index],
            labels[train_index],
            calibration.iloc[train_index]["query_clean"].tolist(),
            calibration.iloc[valid_index]["query_clean"].tolist(),
            articles,
            link_graph,
        )
        if not result:
            result = {
                name: np.zeros_like(labels, dtype=np.float32) for name in fold_scores
            }
        for name, values in fold_scores.items():
            result[name][valid_index] = values
    return result


def _build_test_query_scores(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    query_space: QueryFeatureSpace,
    articles: pd.DataFrame,
    link_graph: np.ndarray,
) -> dict[str, np.ndarray]:
    """Train supervised query models on all calibration rows."""
    return score_query_sources(
        query_space.calibration(),
        query_space.calibration_char(),
        query_space.transform(test["query_clean"]),
        query_space.transform_char(test["query_clean"]),
        labels,
        calibration["query_clean"].tolist(),
        test["query_clean"].tolist(),
        articles,
        link_graph,
    )


def fit_predict(
    data_dir: Path,
) -> pd.DataFrame:
    """Train the complete retrieval system and predict test queries."""
    articles, calibration, test = load_data(data_dir)

    article_ids = articles["article_id"].astype(int).to_numpy()
    labels = build_label_matrix(calibration, article_ids)
    link_graph = _build_link_graph(articles)

    article_index = ArticleFeatureIndex().fit(articles)
    static_calibration = article_index.score(calibration)
    static_test = article_index.score(test)

    query_space = QueryFeatureSpace().fit(calibration["query_clean"])
    query_calibration = _build_oof_query_scores(
        calibration,
        labels,
        query_space,
        articles,
        link_graph,
    )
    query_test = _build_test_query_scores(
        calibration,
        test,
        labels,
        query_space,
        articles,
        link_graph,
    )

    train_scores = {**query_calibration, **static_calibration}
    test_scores = {**query_test, **static_test}
    static_names = set(static_calibration)
    seen_articles = np.flatnonzero(labels.sum(axis=0) > 0)

    train_pairs = build_pair_data(
        np.arange(len(calibration)),
        train_scores,
        _build_metadata(calibration, articles),
        static_names,
        seen_articles,
        labels,
    )
    test_pairs = build_pair_data(
        np.arange(len(test)),
        test_scores,
        _build_metadata(test, articles),
        static_names,
        seen_articles,
    )

    models = [
        fit_lgbm_bagged(train_pairs, 42),
        fit_lgbm_bagged(train_pairs, 123),
        fit_xgboost_pairwise(train_pairs),
        fit_lgbm_stable(train_pairs),
    ]
    model_scores = [
        predict_pairs(model, test_pairs, len(article_ids)) for model in models
    ]
    final_scores = blend_rankers(model_scores)

    answer = make_answer(test["query_id"], final_scores, article_ids)
    validate_answer(answer, test, article_ids)
    return answer
