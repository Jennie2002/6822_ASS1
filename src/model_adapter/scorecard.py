import math

import pandas as pd


SCORE_MIN = 300
SCORE_MAX = 850
BASE_SCORE = 600
BASE_ODDS = 1
POINTS_TO_DOUBLE_ODDS = 50


def probability_to_score(good_probability: float) -> int:
    """Convert good/repaid probability to a conventional 300-850 score."""
    probability = min(max(float(good_probability), 0.001), 0.999)
    odds = probability / (1 - probability)
    factor = POINTS_TO_DOUBLE_ODDS / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    score = offset + factor * math.log(odds)
    return int(round(min(max(score, SCORE_MIN), SCORE_MAX)))


def risk_band(score: int) -> str:
    if score >= 750:
        return "Very low risk"
    if score >= 700:
        return "Low risk"
    if score >= 600:
        return "Medium risk"
    if score >= 500:
        return "High risk"
    return "Very high risk"


def decision_recommendation(score: int) -> str:
    if score >= 700:
        return "Approve"
    if score >= 600:
        return "Manual review"
    return "Reject"


def human_review_trigger(score: int, default_probability: float) -> bool:
    near_boundary = 580 <= score <= 720
    high_default_risk = default_probability >= 0.35
    return near_boundary or high_default_risk


def add_scorecard_outputs(scored: pd.DataFrame) -> pd.DataFrame:
    """Add PD, score, risk band, and decision columns to scored applicants."""
    result = scored.copy()
    result["default_probability"] = 1 - result["approval_probability"]
    result["credit_score"] = result["approval_probability"].apply(probability_to_score)
    result["risk_band"] = result["credit_score"].apply(risk_band)
    result["decision_recommendation"] = result["credit_score"].apply(decision_recommendation)
    result["human_review_trigger"] = result.apply(
        lambda row: human_review_trigger(
            row["credit_score"], row["default_probability"]
        ),
        axis=1,
    )
    return result


REASON_CODE_TEXT = {
    "debt_ratio": "High debt ratio",
    "debt_to_income": "High debt-to-income ratio",
    "late_payment_count": "Recent or repeated late payments",
    "late_payments": "Recent or repeated late payments",
    "NumberOfTimes90DaysLate": "Severe late payment history",
    "monthly_income": "Lower monthly income",
    "income": "Lower income",
    "credit_amount": "Large credit amount",
    "credit_history_years": "Short credit history",
    "employment_years": "Short employment history",
    "savings": "Lower savings buffer",
    "credit_lines": "High number of open credit lines",
    "dependents": "Higher dependent count",
}


LOW_IS_RISKY = {
    "monthly_income",
    "income",
    "credit_history_years",
    "employment_years",
    "savings",
}


def generate_reason_codes(
    applicant: pd.Series,
    reference_data: pd.DataFrame,
    feature_columns: list[str],
    max_reasons: int = 4,
) -> list[str]:
    """Generate simple scorecard-style reason codes for one applicant."""
    reasons = []
    numeric_features = [
        column
        for column in feature_columns
        if column in applicant.index and pd.api.types.is_numeric_dtype(reference_data[column])
    ]

    for feature in numeric_features:
        if feature.startswith("source_"):
            continue
        value = applicant[feature]
        median = reference_data[feature].median()
        if pd.isna(value) or pd.isna(median):
            continue

        is_risky = value < median if feature in LOW_IS_RISKY else value > median
        if not is_risky:
            continue

        distance = abs(value - median) / (reference_data[feature].std() or 1)
        text = REASON_CODE_TEXT.get(feature, f"{feature} differs from portfolio median")
        reasons.append((distance, text))

    reasons = sorted(reasons, key=lambda item: item[0], reverse=True)
    unique_reasons = []
    for _, reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
        if len(unique_reasons) == max_reasons:
            break

    return unique_reasons or ["No major scorecard driver identified"]
