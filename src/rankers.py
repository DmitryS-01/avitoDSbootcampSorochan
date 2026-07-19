"""Pair features, learning-to-rank models and score blending."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from metrics import reciprocal_rank_scores, top_indices


@dataclass(frozen=True)
class PairData:
    """Flat query-article table with query groups."""

    features: np.ndarray
    labels: np.ndarray | None
    groups: list[int]
    candidates: list[np.ndarray]


def build_pair_data(
    row_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    static_names: set[str],
    seen_articles: np.ndarray,
    labels: np.ndarray | None = None,
) -> PairData:
    """Create candidate pairs and ranking features."""
    rank_scores = {
        name: reciprocal_rank_scores(value) for name, value in scores.items()
    }
    z_scores = {
        name: (value - value.mean(axis=1, keepdims=True))
        / (value.std(axis=1, keepdims=True) + 1e-6)
        for name, value in scores.items()
    }

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[int] = []
    candidates: list[np.ndarray] = []

    for local_row, global_row in enumerate(row_indices):
        candidate_set = set(map(int, seen_articles))
        for name, value in scores.items():
            depth = 20 if name in static_names else 30
            candidate_set.update(
                top_indices(value[local_row : local_row + 1], depth)[0]
            )
        if labels is not None:
            candidate_set.update(np.flatnonzero(labels[global_row]))

        columns = np.array(sorted(candidate_set), dtype=int)
        candidates.append(columns)
        groups.append(len(columns))

        features: list[np.ndarray] = []
        for name in scores:
            features.extend(
                [
                    scores[name][local_row, columns],
                    rank_scores[name][local_row, columns],
                    z_scores[name][local_row, columns],
                ]
            )

        rank_matrix = np.column_stack(
            [rank_scores[name][local_row, columns] for name in scores]
        )
        features.extend(
            [
                rank_matrix.mean(axis=1),
                rank_matrix.max(axis=1),
                rank_matrix.std(axis=1),
                (rank_matrix >= 0.1).sum(axis=1),
                (rank_matrix >= 0.05).sum(axis=1),
            ]
        )
        features.extend(metadata[name][global_row, columns] for name in metadata)
        rows.append(np.column_stack(features))
        if labels is not None:
            targets.append(labels[global_row, columns])

    return PairData(
        features=np.vstack(rows).astype(np.float32),
        labels=np.concatenate(targets).astype(np.float32) if targets else None,
        groups=groups,
        candidates=candidates,
    )


def fit_lgbm_bagged(dataset: PairData, seed: int) -> lgb.LGBMRanker:
    """Fit the strongest bagged LambdaMART configuration."""
    if dataset.labels is None:
        raise ValueError("Training labels are required")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=450,
        learning_rate=0.015,
        num_leaves=15,
        max_depth=5,
        min_child_samples=15,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        reg_alpha=0.2,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(dataset.features, dataset.labels, group=dataset.groups)
    return model


def fit_lgbm_stable(dataset: PairData) -> lgb.LGBMRanker:
    """Fit a stable LambdaMART model without row bagging."""
    if dataset.labels is None:
        raise ValueError("Training labels are required")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=450,
        learning_rate=0.015,
        num_leaves=15,
        max_depth=5,
        min_child_samples=15,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(dataset.features, dataset.labels, group=dataset.groups)
    return model


def fit_xgboost_pairwise(dataset: PairData) -> XGBRanker:
    """Fit the pairwise XGBoost ranker."""
    if dataset.labels is None:
        raise ValueError("Training labels are required")
    query_ids = np.repeat(np.arange(len(dataset.groups)), dataset.groups)
    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        n_jobs=2,
        random_state=42,
    )
    model.fit(dataset.features, dataset.labels, qid=query_ids, verbose=False)
    return model


def predict_pairs(
    model: object,
    dataset: PairData,
    article_count: int,
) -> np.ndarray:
    """Restore flat pair predictions to a dense matrix."""
    if isinstance(model, lgb.LGBMRanker):
        features = pd.DataFrame(dataset.features, columns=model.feature_name_)
        values = model.predict(features)
    else:
        values = model.predict(dataset.features)
    prediction = np.asarray(values, dtype=np.float32)
    scores = np.full(
        (len(dataset.groups), article_count),
        -1_000_000.0,
        dtype=np.float32,
    )
    offset = 0
    for row, columns in enumerate(dataset.candidates):
        size = len(columns)
        scores[row, columns] = prediction[offset : offset + size]
        offset += size
    return scores


def blend_rankers(score_matrices: list[np.ndarray]) -> np.ndarray:
    """Blend four complementary rankers by reciprocal ranks."""
    weights = np.array(
        [0.46440025, 0.01076991, 0.47668782, 0.04814202],
        dtype=np.float32,
    )
    return sum(
        weight * reciprocal_rank_scores(scores)
        for weight, scores in zip(weights, score_matrices, strict=True)
    )
