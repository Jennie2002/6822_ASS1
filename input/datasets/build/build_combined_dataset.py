from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from input.datasets.build.data_fusion import (
    COMBINED_CREDIT_PATH,
    build_combined_credit_dataset,
)


if __name__ == "__main__":
    data = build_combined_credit_dataset()
    print(f"Saved {COMBINED_CREDIT_PATH}")
    print(f"Rows: {len(data)}")
    print("Sources:")
    print(data["source_dataset"].value_counts().to_string())
