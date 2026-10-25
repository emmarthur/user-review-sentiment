"""Classify one review from stdin using the shared training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sentiment_core import get_or_train_pipeline  # noqa: E402


def main() -> None:
    print("Review sentiment (Doc2Vec + k-NN). Training or loading model…")
    pipeline = get_or_train_pipeline()
    if pipeline.cv_mean_accuracy is not None:
        print(f"Cross-val accuracy (5-fold): {pipeline.cv_mean_accuracy}%")
    print("Enter a review (blank line to quit):\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        result = pipeline.predict(line)
        print(f"{result['sentiment'].upper()} ({result['confidence']}% confidence)\n")


if __name__ == "__main__":
    main()
