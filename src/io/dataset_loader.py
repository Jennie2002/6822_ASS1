from dataclasses import dataclass
from pathlib import Path

import pandas as pd


COMBINED_CREDIT_PATH = Path("input/datasets/processed/combined_credit_dataset.csv")

FAIRNESS_GROUPS = ["age_group", "income_group", "synthetic_protected_group"]
BISG_INPUT_COLUMNS = [
    "surname",
    "postcode",
    "state",
    "geography",
    "census_tract",
    "surname_prob_group_a",
    "surname_prob_group_b",
    "tract_prob_group_a",
    "tract_prob_group_b",
    "bisg_prob_group_a",
    "bisg_prob_group_b",
]


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    feature_columns: list[str]
    fairness_groups: list[str]
    message: str
    warning: str | None = None


def load_combined_credit_dataset(
    path: Path = COMBINED_CREDIT_PATH,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {path}. "
            "Run `python input/datasets/build/build_combined_dataset.py` first."
        )

    raw = pd.read_csv(path)
    if "synthetic_protected_group" not in raw.columns and "protected_group" in raw.columns:
        raw = raw.rename(columns={"protected_group": "synthetic_protected_group"})
    encoded = pd.get_dummies(
        raw,
        columns=[
            "source_dataset",
            "fusion_type",
            "housing",
            "job",
            "saving_accounts",
            "checking_account",
            "purpose",
        ],
        prefix=["source", "fusion", "housing", "job", "savings", "checking", "purpose"],
        dtype=int,
    )

    excluded_columns = {
        "sex",
        "synthetic_protected_group",
        "age_group",
        "income_group",
        "risk_bucket",
        "repaid",
        *BISG_INPUT_COLUMNS,
    }
    feature_columns = [
        column for column in encoded.columns if column not in excluded_columns
    ]

    model_frame = encoded[
        feature_columns
        + ["synthetic_protected_group", "age_group", "income_group", "repaid"]
    ].copy()
    for column in BISG_INPUT_COLUMNS:
        if column in raw.columns:
            model_frame[column] = raw[column]

    model_frame[feature_columns] = model_frame[feature_columns].fillna(0)
    return model_frame, feature_columns, FAIRNESS_GROUPS


def load_dataset(
    data_source: str = "Combined benchmark governance dataset",
    n_applicants: int = 0,
    seed: int = 42,
) -> DatasetBundle:
    _ = (data_source, n_applicants, seed)
    data, feature_columns, fairness_groups = load_combined_credit_dataset()
    return DatasetBundle(
        data=data,
        feature_columns=feature_columns,
        fairness_groups=fairness_groups,
        message=(
            "Using the fixed benchmark governance dataset from "
            f"`{COMBINED_CREDIT_PATH}`."
        ),
    )
