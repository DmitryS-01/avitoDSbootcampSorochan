"""Data loading and preprocessing."""

from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc

from text import ArticleSpellCorrector, prepare_articles, prepare_queries


def read_feather(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Feather V2 file through the Arrow IPC reader."""
    table = ipc.open_file(path).read_all()
    if columns is not None:
        table = table.select(columns)
    return table.to_pandas()


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and preprocess the source Feather files."""
    articles = read_feather(data_dir / "articles.f")
    calibration = read_feather(data_dir / "calibration.f")
    test = read_feather(data_dir / "test.f")
    articles = prepare_articles(articles)
    corrector = ArticleSpellCorrector(articles)
    calibration = prepare_queries(calibration, corrector)
    test = prepare_queries(test, corrector)
    return articles, calibration, test
