from pathlib import Path
import sys

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_RANDOM_SEED, MODEL_TYPES, get_model_profile
from src.io.dataset_loader import load_dataset
from src.model_adapter.credit_model import (
    MODEL_ARTIFACT_DIR,
    model_artifact_filename,
    save_model_artifact,
    train_model_artifact,
)


def build_model_artifacts(seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    dataset = load_dataset(seed=seed)
    train_data, _ = train_test_split(
        dataset.data,
        test_size=0.25,
        random_state=seed,
        stratify=dataset.data["repaid"],
    )

    rows = []
    for model_type in MODEL_TYPES:
        model = train_model_artifact(train_data, dataset.feature_columns, model_type, seed)
        artifact_path = MODEL_ARTIFACT_DIR / model_artifact_filename(model_type)
        save_model_artifact(model, artifact_path)

        profile = get_model_profile(model_type)
        rows.append(
            {
                "model_id": model_artifact_filename(model_type).replace(".pkl", ""),
                "model_type": model_type,
                "role": profile["role"],
                "complexity_level": profile["complexity_level"],
                "explainability_risk": profile["explainability_risk"],
                "deployment_gate": profile["deployment_gate"],
                "artifact_path": str(artifact_path),
                "training_dataset": "input/datasets/processed/combined_credit_dataset.csv",
                "training_seed": seed,
                "sklearn_version": sklearn.__version__,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MODEL_ARTIFACT_DIR / "model_manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    manifest = build_model_artifacts()
    print("Saved model artifacts:")
    print(manifest[["model_type", "artifact_path"]].to_string(index=False))
