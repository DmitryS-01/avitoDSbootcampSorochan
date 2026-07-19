"""Popularity baseline for help-center article retrieval."""

from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc
from sklearn.model_selection import KFold

DATA_DIR = Path("statement/candidate_public/candidate_data")
OUTPUT_PATH = Path("outputs/answer.csv")
TOP_K = 10
N_SPLITS = 5


def read_feather(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from a Feather V2 file."""
    return ipc.open_file(path).read_all().select(columns).to_pandas()


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read articles, calibration queries and test queries."""
    articles = read_feather(DATA_DIR / "articles.f", ["article_id"])
    calibration = read_feather(
        DATA_DIR / "calibration.f",
        ["ground_truth"],
    )
    test = read_feather(DATA_DIR / "test.f", ["query_id"])
    return articles, calibration, test


def parse_ground_truth(value: str) -> set[int]:
    """Convert a space-separated ground-truth string to article IDs."""
    return {int(article_id) for article_id in value.split()}


def build_popularity_ranking(
    calibration: pd.DataFrame,
    article_ids: pd.Series,
) -> list[int]:
    """Rank articles by their frequency in the calibration labels."""
    popularity = (
        calibration["ground_truth"].str.split().explode().astype("int64").value_counts()
    )
    ranking = pd.DataFrame({"article_id": article_ids.astype("int64")})
    ranking["frequency"] = ranking["article_id"].map(popularity).fillna(0)
    ranking = ranking.sort_values(
        ["frequency", "article_id"],
        ascending=[False, True],
    )
    return ranking["article_id"].head(TOP_K).tolist()


def average_precision_at_k(
    ranking: list[int],
    relevant: set[int],
) -> float:
    """Calculate average precision for one query."""
    hits = 0
    precision_sum = 0.0
    for rank, article_id in enumerate(ranking, start=1):
        if article_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), TOP_K)


def evaluate_popularity(
    calibration: pd.DataFrame,
    article_ids: pd.Series,
) -> float:
    """Evaluate popularity rankings out of fold."""
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = []

    for train_indices, valid_indices in splitter.split(calibration):
        train = calibration.iloc[train_indices]
        valid = calibration.iloc[valid_indices]
        ranking = build_popularity_ranking(train, article_ids)
        scores.extend(
            average_precision_at_k(ranking, parse_ground_truth(ground_truth))
            for ground_truth in valid["ground_truth"]
        )
    return sum(scores) / len(scores)


def make_answer(test: pd.DataFrame, ranking: list[int]) -> pd.DataFrame:
    """Build an answer.csv-compatible dataframe."""
    answer = test[["query_id"]].copy()
    answer["answer"] = " ".join(map(str, ranking))
    return answer


def validate_answer(
    answer: pd.DataFrame,
    test: pd.DataFrame,
    article_ids: pd.Series,
) -> None:
    """Validate submission columns, query order and article IDs."""
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Expected columns: query_id, answer")
    if not answer["query_id"].equals(test["query_id"]):
        raise ValueError("query_id order does not match test.f")

    valid_ids = set(article_ids.astype(int))
    for value in answer["answer"]:
        predictions = [int(article_id) for article_id in value.split()]
        if len(predictions) != TOP_K:
            raise ValueError(f"Each answer must contain {TOP_K} IDs")
        if len(predictions) != len(set(predictions)):
            raise ValueError("Duplicate article_id in one answer")
        if not set(predictions).issubset(valid_ids):
            raise ValueError("Unknown article_id in answer")


def main() -> None:
    """Evaluate the baseline and write predictions for the test set."""
    articles, calibration, test = read_data()
    score = evaluate_popularity(calibration, articles["article_id"])
    ranking = build_popularity_ranking(calibration, articles["article_id"])

    answer = make_answer(test, ranking)
    validate_answer(answer, test, articles["article_id"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(OUTPUT_PATH, index=False)

    print(f"Popularity baseline MAP@{TOP_K}: {score:.4f}")
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
