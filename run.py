"""Generate the final submission."""

import sys
from pathlib import Path

DATA_DIR = Path("statement/candidate_public/candidate_data")
OUTPUT_PATH = Path("outputs/answer.csv")
sys.path.insert(0, "src")

from pipeline import fit_predict  # noqa: E402


def main() -> None:
    """Train the retrieval pipeline and save predictions."""
    answer = fit_predict(DATA_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(answer)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
