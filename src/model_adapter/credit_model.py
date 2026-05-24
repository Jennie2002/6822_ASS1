import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


MODEL_ARTIFACT_DIR = Path("input/model_artifacts")
MODEL_MANIFEST_PATH = MODEL_ARTIFACT_DIR / "model_manifest.csv"


def build_classifier(model_type: str, seed: int) -> Pipeline:
    if model_type == "Logistic Regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

    if model_type == "XGBoost":
        classifier = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", classifier),
            ]
        )

    if model_type == "FNN":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(24, 12),
                        activation="relu",
                        alpha=0.01,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def train_credit_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    test_size: float,
    seed: int,
) -> tuple[object, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data[feature_columns]
    y = data["repaid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = build_classifier(model_type, seed)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def train_model_artifact(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    seed: int,
) -> object:
    model = build_classifier(model_type, seed)
    model.fit(data[feature_columns], data["repaid"])
    return model


def model_artifact_filename(model_type: str) -> str:
    return model_type.lower().replace(" ", "_") + ".pkl"


def save_model_artifact(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model_artifact(path: Path) -> object:
    with path.open("rb") as file:
        model = pickle.load(file)
    return repair_loaded_model(model)


def repair_loaded_model(model: object) -> object:
    """Patch small sklearn pickle compatibility gaps for prototype artifacts."""
    classifier = (
        model.named_steps.get("classifier")
        if hasattr(model, "named_steps") and "classifier" in model.named_steps
        else model
    )

    if isinstance(classifier, LogisticRegression) and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"

    return model


def load_model_manifest(path: Path = MODEL_MANIFEST_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model manifest: {path}. "
            "Run `python input/model_artifacts/build_model_artifacts.py` first."
        )
    return pd.read_csv(path)


def load_registered_model(model_type: str) -> object:
    manifest = load_model_manifest()
    matches = manifest[manifest["model_type"] == model_type]
    if matches.empty:
        raise ValueError(f"Model artifact is not registered: {model_type}")
    artifact_path = Path(matches.iloc[0]["artifact_path"])
    return load_model_artifact(artifact_path)


def add_predictions(
    model: object,
    data: pd.DataFrame,
    feature_columns: list[str],
    decision_threshold: float,
) -> pd.DataFrame:
    scored = data.copy()
    scored["approval_probability"] = model.predict_proba(scored[feature_columns])[:, 1]
    scored["approved"] = (scored["approval_probability"] >= decision_threshold).astype(int)
    return scored


def feature_importance(
    model: object, feature_columns: list[str], model_type: str
) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        coefficients = model.named_steps["classifier"].coef_[0]
        values = np.abs(coefficients)
        label = "absolute_coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        label = "importance"
    elif hasattr(classifier, "coefs_"):
        values = np.abs(classifier.coefs_[0]).mean(axis=1)
        label = "first_layer_weight"
    else:
        values = np.zeros(len(feature_columns))
        label = "importance"

    importance = pd.DataFrame({"feature": feature_columns, label: values})
    return importance.sort_values(label, ascending=False)
