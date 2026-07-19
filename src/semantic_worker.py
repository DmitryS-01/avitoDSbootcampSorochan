"""Isolated entry point for neural retrieval models."""

import sys
from pathlib import Path

import numpy as np

from data import load_data
from metrics import build_label_matrix
from semantic import cross_encoder_scores, semantic_query_scores


def main() -> None:
    """Run one neural scoring stage and save its matrix."""
    mode = sys.argv[1]
    data_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    articles, calibration, test = load_data(data_dir)

    if mode == "semantic":
        article_ids = articles["article_id"].astype(int).to_numpy()
        labels = build_label_matrix(calibration, article_ids)
        scores = semantic_query_scores(calibration, test, labels)
    elif mode == "cross":
        candidates = np.load(sys.argv[4])
        scores = cross_encoder_scores(test, articles, candidates)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    np.savez_compressed(output_path, scores=scores)


if __name__ == "__main__":
    main()
