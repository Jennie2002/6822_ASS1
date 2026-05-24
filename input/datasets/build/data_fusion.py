from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


GIVE_ME_SOME_CREDIT_PATH = Path("input/datasets/raw/give_me_some_credit_data.csv")
GERMAN_CREDIT_PATH = Path("input/datasets/raw/german_credit_data.csv")
COMBINED_CREDIT_PATH = Path("input/datasets/processed/combined_credit_dataset.csv")

GMSC_SAMPLE_SIZE = 20_000
FUSED_SAMPLE_SIZE = 5_000
RANDOM_SEED = 42

NUMERIC_COLUMNS = [
    "age",
    "monthly_income",
    "debt_ratio",
    "credit_amount",
    "loan_duration_months",
    "late_payment_count",
    "credit_lines",
    "dependents",
]

CATEGORICAL_COLUMNS = [
    "sex",
    "housing",
    "job",
    "saving_accounts",
    "checking_account",
    "purpose",
    "fusion_type",
]

COMBINED_BASE_COLUMNS = [
    "source_dataset",
    "fusion_type",
    *NUMERIC_COLUMNS,
    *CATEGORICAL_COLUMNS[:-1],
    "synthetic_protected_group",
    "age_group",
    "income_group",
    "risk_bucket",
    "repaid",
]

COMBINED_FAIRNESS_GROUPS = ["age_group", "income_group", "synthetic_protected_group"]
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


def _base_proxy_probability(data: pd.DataFrame) -> np.ndarray:
    if "synthetic_protected_group" in data.columns:
        return np.where(data["synthetic_protected_group"].astype(str) == "female", 0.46, 0.54)
    if "age_group" in data.columns:
        return np.where(data["age_group"].astype(str) == "Under 35", 0.44, 0.60)
    return np.repeat(0.50, len(data))


def _normalised_pair(probability_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability_a = np.clip(probability_a, 0.03, 0.97)
    probability_b = 1 - probability_a
    return probability_a, probability_b


def add_faker_bisg_data(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    if all(column in data.columns for column in BISG_INPUT_COLUMNS):
        return data

    fake = Faker("en_US")
    fake.seed_instance(seed)
    rng = np.random.default_rng(seed)

    enriched = data.copy()
    n_rows = len(enriched)
    states = [fake.state_abbr() for _ in range(n_rows)]

    enriched["surname"] = [fake.last_name() for _ in range(n_rows)]
    enriched["postcode"] = [fake.postcode() for _ in range(n_rows)]
    enriched["state"] = states
    enriched["geography"] = [fake.city() for _ in range(n_rows)]
    enriched["census_tract"] = [
        f"{state}-{rng.integers(100000, 999999)}" for state in states
    ]

    base_probability = _base_proxy_probability(enriched)
    surname_prob_a, surname_prob_b = _normalised_pair(
        base_probability + rng.normal(0, 0.10, n_rows)
    )
    tract_prob_a, tract_prob_b = _normalised_pair(
        base_probability + rng.normal(0, 0.12, n_rows)
    )

    prior_a = float(np.mean(base_probability))
    prior_b = 1 - prior_a
    bayes_score_a = (surname_prob_a * tract_prob_a) / max(prior_a, 0.01)
    bayes_score_b = (surname_prob_b * tract_prob_b) / max(prior_b, 0.01)
    denominator = bayes_score_a + bayes_score_b

    enriched["surname_prob_group_a"] = surname_prob_a
    enriched["surname_prob_group_b"] = surname_prob_b
    enriched["tract_prob_group_a"] = tract_prob_a
    enriched["tract_prob_group_b"] = tract_prob_b
    enriched["bisg_prob_group_a"] = bayes_score_a / denominator
    enriched["bisg_prob_group_b"] = bayes_score_b / denominator
    return enriched


def age_group(age: float) -> str:
    return "Under 35" if age < 35 else "35 and over"


def income_group(monthly_income: float) -> str:
    if pd.isna(monthly_income):
        return "Unknown income"
    return "Lower income" if monthly_income < 5000 else "Higher income"


def gmsc_risk_bucket(repaid: pd.Series, debt_ratio: pd.Series, late_count: pd.Series) -> pd.Series:
    bucket = np.where(repaid == 0, "observed_bad", "observed_good")
    bucket = np.where((repaid == 1) & ((debt_ratio > 0.6) | (late_count > 0)), "good_watch", bucket)
    return pd.Series(bucket, index=repaid.index)


def german_risk_bucket(repaid: pd.Series, credit_amount: pd.Series, duration: pd.Series) -> pd.Series:
    bucket = np.where(repaid == 0, "observed_bad", "observed_good")
    bucket = np.where((repaid == 1) & ((credit_amount > 5000) | (duration > 36)), "good_watch", bucket)
    return pd.Series(bucket, index=repaid.index)


def normalise_give_me_some_credit(path: Path = GIVE_ME_SOME_CREDIT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Give Me Some Credit file: {path}")

    raw = pd.read_csv(path)
    raw = raw.drop(columns=[col for col in raw.columns if col.lower().startswith("unnamed")])
    required = {
        "SeriousDlqin2yrs",
        "age",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing Give Me Some Credit columns: {', '.join(sorted(missing))}")

    late_payment_count = (
        raw["NumberOfTime30-59DaysPastDueNotWorse"]
        + raw["NumberOfTimes90DaysLate"]
        + raw["NumberOfTime60-89DaysPastDueNotWorse"]
    )
    repaid = 1 - raw["SeriousDlqin2yrs"]

    data = pd.DataFrame(
        {
            "source_dataset": "give_me_some_credit",
            "fusion_type": "gmsc_observed_behavior_enriched_context",
            "age": raw["age"],
            "monthly_income": raw["MonthlyIncome"],
            "debt_ratio": raw["DebtRatio"],
            "credit_amount": np.nan,
            "loan_duration_months": np.nan,
            "late_payment_count": late_payment_count,
            "credit_lines": raw["NumberOfOpenCreditLinesAndLoans"]
            + raw["NumberRealEstateLoansOrLines"],
            "dependents": raw["NumberOfDependents"],
            "sex": np.nan,
            "housing": np.nan,
            "job": np.nan,
            "saving_accounts": np.nan,
            "checking_account": np.nan,
            "purpose": np.nan,
            "synthetic_protected_group": "not_available",
            "age_group": raw["age"].apply(age_group),
            "income_group": raw["MonthlyIncome"].apply(income_group),
            "risk_bucket": gmsc_risk_bucket(repaid, raw["DebtRatio"], late_payment_count),
            "repaid": repaid,
        }
    )
    return data


def normalise_german_credit(path: Path = GERMAN_CREDIT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing German Credit file: {path}")

    raw = pd.read_csv(path)
    raw = raw.drop(columns=[col for col in raw.columns if col.lower().startswith("unnamed")])
    raw = raw.rename(
        columns={
            "Age": "age",
            "Sex": "sex",
            "Job": "job",
            "Housing": "housing",
            "Saving accounts": "saving_accounts",
            "Checking account": "checking_account",
            "Credit amount": "credit_amount",
            "Duration": "duration",
            "Purpose": "purpose",
            "Risk": "risk",
        }
    )
    required = {
        "age",
        "sex",
        "job",
        "housing",
        "saving_accounts",
        "checking_account",
        "credit_amount",
        "duration",
        "purpose",
        "risk",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing German Credit columns: {', '.join(sorted(missing))}")

    repaid = raw["risk"].map({"good": 1, "bad": 0})
    data = pd.DataFrame(
        {
            "source_dataset": "german_credit",
            "fusion_type": "german_observed_context_enriched_behavior",
            "age": raw["age"],
            "monthly_income": np.nan,
            "debt_ratio": np.nan,
            "credit_amount": raw["credit_amount"],
            "loan_duration_months": raw["duration"],
            "late_payment_count": np.nan,
            "credit_lines": np.nan,
            "dependents": np.nan,
            "sex": raw["sex"].fillna("unknown").astype(str),
            "housing": raw["housing"].fillna("unknown").astype(str),
            "job": raw["job"].fillna("unknown").astype(str),
            "saving_accounts": raw["saving_accounts"].fillna("unknown").astype(str),
            "checking_account": raw["checking_account"].fillna("unknown").astype(str),
            "purpose": raw["purpose"].fillna("unknown").astype(str),
            "synthetic_protected_group": raw["sex"].fillna("unknown").astype(str),
            "age_group": raw["age"].apply(age_group),
            "income_group": "Unknown income",
            "risk_bucket": german_risk_bucket(repaid, raw["credit_amount"], raw["duration"]),
            "repaid": repaid,
        }
    )
    return data


def sample_like(
    target: pd.DataFrame,
    donor_pool: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    donor_groups = {
        key: frame.reset_index(drop=True)
        for key, frame in donor_pool.groupby(["age_group", "repaid", "risk_bucket"])
    }
    fallback_repaid = {
        key: frame.reset_index(drop=True) for key, frame in donor_pool.groupby("repaid")
    }
    all_donors = donor_pool.reset_index(drop=True)
    rows = []

    for _, row in target.iterrows():
        key = (row["age_group"], row["repaid"], row["risk_bucket"])
        candidates = donor_groups.get(key)
        if candidates is None or candidates.empty:
            candidates = fallback_repaid.get(row["repaid"], all_donors)
        rows.append(candidates.iloc[int(rng.integers(0, len(candidates)))])

    return pd.DataFrame(rows).reset_index(drop=True)


def jitter_numeric(values: pd.Series, rng: np.random.Generator, sigma: float = 0.12) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    noise = rng.lognormal(mean=0, sigma=sigma, size=len(numeric))
    return np.maximum(numeric * noise, 0)


def enrich_gmsc_with_german_context(
    gmsc: pd.DataFrame, german: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    enriched = gmsc.reset_index(drop=True).copy()
    donors = sample_like(enriched, german, rng)
    for column in ["sex", "housing", "job", "saving_accounts", "checking_account", "purpose"]:
        enriched[column] = donors[column].to_numpy()
    enriched["synthetic_protected_group"] = donors["sex"].to_numpy()
    enriched["credit_amount"] = jitter_numeric(donors["credit_amount"], rng, sigma=0.18)
    enriched["loan_duration_months"] = np.rint(jitter_numeric(donors["loan_duration_months"], rng, sigma=0.08))
    return enriched


def enrich_german_with_gmsc_behavior(
    german: pd.DataFrame, gmsc: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    enriched = german.reset_index(drop=True).copy()
    donors = sample_like(enriched, gmsc, rng)
    for column in ["monthly_income", "debt_ratio", "late_payment_count", "credit_lines", "dependents"]:
        sigma = 0.10 if column in {"monthly_income", "debt_ratio"} else 0.18
        enriched[column] = jitter_numeric(donors[column], rng, sigma=sigma)
    enriched["late_payment_count"] = np.rint(enriched["late_payment_count"])
    enriched["credit_lines"] = np.rint(enriched["credit_lines"])
    enriched["dependents"] = np.rint(enriched["dependents"])
    enriched["income_group"] = enriched["monthly_income"].apply(income_group)
    return enriched


def build_interpolated_rows(
    gmsc: pd.DataFrame,
    german: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = gmsc.sample(n=sample_size, replace=len(gmsc) < sample_size, random_state=RANDOM_SEED)
    base = base.reset_index(drop=True)
    donors = sample_like(base, german, rng)
    fused = base.copy()
    fused["source_dataset"] = "conditional_fusion"
    fused["fusion_type"] = "conditional_donor_interpolation"
    for column in ["sex", "housing", "job", "saving_accounts", "checking_account", "purpose"]:
        fused[column] = donors[column].to_numpy()
    fused["synthetic_protected_group"] = donors["sex"].to_numpy()
    fused["credit_amount"] = (
        0.55 * jitter_numeric(donors["credit_amount"], rng, sigma=0.12)
        + 0.45 * np.maximum(fused["monthly_income"].fillna(0).to_numpy() * fused["debt_ratio"].fillna(0).to_numpy(), 0)
    )
    fused["loan_duration_months"] = np.rint(jitter_numeric(donors["loan_duration_months"], rng, sigma=0.08))
    return fused


def fuse_with_conditional_donor_matching(
    gmsc: pd.DataFrame,
    german: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gmsc_sample = gmsc.sample(
        n=min(GMSC_SAMPLE_SIZE, len(gmsc)),
        random_state=seed,
    ).reset_index(drop=True)
    german = german.reset_index(drop=True)

    enriched_gmsc = enrich_gmsc_with_german_context(gmsc_sample, german, rng)
    enriched_german = enrich_german_with_gmsc_behavior(german, gmsc_sample, rng)
    fused_rows = build_interpolated_rows(gmsc_sample, german, FUSED_SAMPLE_SIZE, rng)

    combined = pd.concat([enriched_gmsc, enriched_german, fused_rows], ignore_index=True)
    combined = combined[COMBINED_BASE_COLUMNS]

    for column in NUMERIC_COLUMNS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
        source_median = combined.groupby("source_dataset")[column].transform("median")
        overall_median = combined[column].median()
        combined[column] = combined[column].fillna(source_median).fillna(overall_median).fillna(0)

    for column in CATEGORICAL_COLUMNS:
        combined[column] = combined[column].fillna("unknown").astype(str)

    combined["repaid"] = pd.to_numeric(combined["repaid"], errors="coerce")
    combined = combined.dropna(subset=["repaid"])
    combined["repaid"] = combined["repaid"].astype(int)
    combined["age"] = combined["age"].clip(lower=18, upper=100)
    combined["loan_duration_months"] = combined["loan_duration_months"].clip(lower=1, upper=120)
    combined["debt_ratio"] = combined["debt_ratio"].clip(lower=0, upper=5)
    combined["late_payment_count"] = combined["late_payment_count"].clip(lower=0)
    combined["credit_lines"] = combined["credit_lines"].clip(lower=0)
    combined["dependents"] = combined["dependents"].clip(lower=0)
    combined["income_group"] = combined["monthly_income"].apply(income_group)
    return combined


def build_combined_credit_dataset(output_path: Path = COMBINED_CREDIT_PATH) -> pd.DataFrame:
    gmsc = normalise_give_me_some_credit()
    german = normalise_german_credit()
    combined = fuse_with_conditional_donor_matching(gmsc, german)
    combined = add_faker_bisg_data(combined, seed=RANDOM_SEED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined
