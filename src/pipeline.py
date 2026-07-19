"""End-to-end training and prediction pipeline."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import load_data
from features import ArticleFeatureIndex, QueryFeatureSpace
from metrics import (
    build_label_matrix,
    make_answer,
    top_indices,
    validate_answer,
)
from query_models import score_query_sources
from rankers import (
    blend_base_rankers,
    blend_rankers,
    build_pair_data,
    fit_automl_xgboost,
    fit_lgbm_bagged,
    predict_pairs,
)


def _semantic_worker(
    data_dir: Path,
    mode: str,
    output_path: Path,
    candidates_path: Path | None = None,
) -> np.ndarray:
    """Run neural inference outside the tabular-model process."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("semantic_worker.py")),
        mode,
        str(data_dir),
        str(output_path),
    ]
    if candidates_path is not None:
        command.append(str(candidates_path))
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    subprocess.run(command, check=True, env=environment)
    with np.load(output_path) as saved:
        return saved["scores"]


def _build_link_graph(
    articles: pd.DataFrame,
) -> np.ndarray:
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


def _build_fold_local_oof_query_scores(
    calibration: pd.DataFrame,
    labels: np.ndarray,
    articles: pd.DataFrame,
    link_graph: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build strictly fold-local query features for the noise model."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    result: dict[str, np.ndarray] = {}
    for train_index, valid_index in splitter.split(calibration):
        train_queries = calibration.iloc[train_index]
        valid_queries = calibration.iloc[valid_index]
        fold_space = QueryFeatureSpace().fit(train_queries["query_clean"])
        fold_scores = score_query_sources(
            fold_space.calibration(),
            fold_space.calibration_char(),
            fold_space.transform(valid_queries["query_clean"]),
            fold_space.transform_char(valid_queries["query_clean"]),
            labels[train_index],
            train_queries["query_clean"].tolist(),
            valid_queries["query_clean"].tolist(),
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
    temporary = tempfile.TemporaryDirectory(prefix="avito-semantic-")
    temporary_dir = Path(temporary.name)
    semantic_path = temporary_dir / "semantic.npz"
    semantic_scores = _semantic_worker(
        data_dir,
        "semantic",
        semantic_path,
    )
    link_graph = _build_link_graph(articles)

    article_index = ArticleFeatureIndex().fit(articles)
    static_calibration = article_index.score(calibration)
    static_test = article_index.score(test)
    noise_static_calibration = article_index.score_noise_spell(calibration)
    noise_static_test = article_index.score_noise_spell(test)

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
    noise_query_calibration = _build_fold_local_oof_query_scores(
        calibration,
        labels,
        articles,
        link_graph,
    )
    noise_query_test = _build_test_query_scores(
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
    noise_train_scores = {
        **noise_query_calibration,
        **static_calibration,
        **noise_static_calibration,
    }
    noise_test_scores = {
        **noise_query_test,
        **static_test,
        **noise_static_test,
    }
    noise_static_names = {
        *static_calibration,
        *noise_static_calibration,
    }
    seen_articles = np.flatnonzero(labels.sum(axis=0) > 0)
    calibration_metadata = _build_metadata(calibration, articles)
    test_metadata = _build_metadata(test, articles)

    train_pairs = build_pair_data(
        np.arange(len(calibration)),
        train_scores,
        calibration_metadata,
        static_names,
        seen_articles,
        labels,
    )
    test_pairs = build_pair_data(
        np.arange(len(test)),
        test_scores,
        test_metadata,
        static_names,
        seen_articles,
    )
    noise_train_pairs = build_pair_data(
        np.arange(len(calibration)),
        noise_train_scores,
        calibration_metadata,
        noise_static_names,
        seen_articles,
        labels,
    )
    noise_test_pairs = build_pair_data(
        np.arange(len(test)),
        noise_test_scores,
        test_metadata,
        noise_static_names,
        seen_articles,
    )

    models = [
        fit_lgbm_bagged(train_pairs, 42),
        fit_lgbm_bagged(train_pairs, 123),
    ]
    model_scores = [
        predict_pairs(model, test_pairs, len(article_ids)) for model in models
    ]
    base_scores = blend_base_rankers(model_scores, semantic_scores)
    automl_model = fit_automl_xgboost(noise_train_pairs)
    automl_scores = predict_pairs(
        automl_model,
        noise_test_pairs,
        len(article_ids),
    )
    candidates_path = temporary_dir / "candidates.npy"
    np.save(candidates_path, top_indices(base_scores, 10))
    cross_scores = _semantic_worker(
        data_dir,
        "cross",
        temporary_dir / "cross.npz",
        candidates_path,
    )
    final_scores = blend_rankers(base_scores, automl_scores, cross_scores)

    answer = make_answer(test["query_id"], final_scores, article_ids)
    validate_answer(answer, test, article_ids)
    temporary.cleanup()
    return answer
