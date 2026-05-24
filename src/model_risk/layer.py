import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.config import get_model_profile, get_threshold


BISG_PROBABILITY_COLUMNS = ["bisg_prob_group_a", "bisg_prob_group_b"]


def compute_performance(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None
) -> dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["ROC-AUC"] = 0.0
        metrics["KS"] = compute_ks(y_true, y_score)
    return metrics


def compute_ks(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values(
        "y_score", ascending=False
    )
    data["good"] = data["y_true"]
    data["bad"] = 1 - data["y_true"]

    good_total = data["good"].sum()
    bad_total = data["bad"].sum()
    if good_total == 0 or bad_total == 0:
        return 0.0

    data["cum_good"] = data["good"].cumsum() / good_total
    data["cum_bad"] = data["bad"].cumsum() / bad_total
    return float((data["cum_good"] - data["cum_bad"]).abs().max())


def compute_fairness(scored: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, float]:
    rows = []

    for group_name, group_data in scored.groupby(group_column):
        negatives = group_data[group_data["repaid"] == 0]
        positives = group_data[group_data["repaid"] == 1]
        rows.append(
            {
                "group": group_name,
                "count": len(group_data),
                "approval_rate": group_data["approved"].mean(),
                "false_positive_rate": (
                    ((negatives["approved"] == 1).mean()) if len(negatives) else 0
                ),
                "false_negative_rate": (
                    ((positives["approved"] == 0).mean()) if len(positives) else 0
                ),
            }
        )

    fairness = pd.DataFrame(rows).sort_values("group")
    approval_disparity = (
        fairness["approval_rate"].max() - fairness["approval_rate"].min()
        if len(fairness)
        else 0
    )
    return fairness, approval_disparity


def fairness_disparity_summary(
    fairness: pd.DataFrame,
    *,
    group_column: str,
    method: str,
    threshold: float,
) -> dict[str, object]:
    approval_disparity = disparity_range(fairness["approval_rate"])
    fpr_disparity = disparity_range(fairness["false_positive_rate"])
    fnr_disparity = disparity_range(fairness["false_negative_rate"])
    status = (
        "Review required"
        if max(approval_disparity, fpr_disparity, fnr_disparity) > threshold
        else "Pass"
    )
    return {
        "group_column": group_column,
        "method": method,
        "approval_disparity": approval_disparity,
        "fpr_disparity": fpr_disparity,
        "fnr_disparity": fnr_disparity,
        "governance_status": status,
    }


def compute_multilayer_fairness(
    scored: pd.DataFrame,
    group_columns: list[str],
    threshold: float,
) -> dict[str, object]:
    details = {}
    rows = []
    for group_column in group_columns:
        if group_column not in scored.columns:
            continue
        fairness, _ = compute_fairness(scored, group_column)
        fairness = fairness.copy()
        fairness.insert(0, "fairness_group", group_column)
        summary = fairness_disparity_summary(
            fairness,
            group_column=group_column,
            method="Explicit fairness-group audit",
            threshold=threshold,
        )
        details[group_column] = fairness
        rows.append(summary)

    summary_table = pd.DataFrame(rows)
    if summary_table.empty:
        return {
            "summary": summary_table,
            "details": details,
            "worst_approval_disparity": 0.0,
            "worst_fpr_disparity": 0.0,
            "worst_fnr_disparity": 0.0,
            "breached_groups": [],
        }

    breached_groups = summary_table.loc[
        summary_table["governance_status"] == "Review required", "group_column"
    ].tolist()
    return {
        "summary": summary_table,
        "details": details,
        "worst_approval_disparity": float(summary_table["approval_disparity"].max()),
        "worst_fpr_disparity": float(summary_table["fpr_disparity"].max()),
        "worst_fnr_disparity": float(summary_table["fnr_disparity"].max()),
        "breached_groups": breached_groups,
    }


def add_demo_bisg_probabilities(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    _ = seed
    missing = [column for column in BISG_PROBABILITY_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "BISG probability columns are missing from the processed dataset: "
            + ", ".join(missing)
        )
    return data


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float((values * weights).sum() / total_weight)


def compute_bisg_weighted_fairness(
    scored: pd.DataFrame,
    probability_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    probability_columns = probability_columns or BISG_PROBABILITY_COLUMNS
    rows = []

    for column in probability_columns:
        weights = scored[column].astype(float)
        negatives = scored["repaid"] == 0
        positives = scored["repaid"] == 1

        group_name = column.replace("bisg_prob_", "BISG ").replace("_", " ").title()
        rows.append(
            {
                "group": group_name,
                "count": weights.sum(),
                "approval_rate": weighted_mean(scored["approved"], weights),
                "false_positive_rate": weighted_mean(
                    scored.loc[negatives, "approved"],
                    weights.loc[negatives],
                ),
                "false_negative_rate": weighted_mean(
                    1 - scored.loc[positives, "approved"],
                    weights.loc[positives],
                ),
            }
        )

    fairness = pd.DataFrame(rows).sort_values("group")
    approval_disparity = (
        fairness["approval_rate"].max() - fairness["approval_rate"].min()
        if len(fairness)
        else 0
    )
    return fairness, approval_disparity


def compute_bisg_governance_audit(
    scored: pd.DataFrame,
    threshold: float,
) -> dict[str, object]:
    fairness, _ = compute_bisg_weighted_fairness(scored)
    fairness = fairness.copy()
    fairness.insert(0, "fairness_group", "BISG soft weighting")
    summary = fairness_disparity_summary(
        fairness,
        group_column="BISG soft weighting",
        method="BISG-weighted supplementary proxy monitoring",
        threshold=threshold,
    )
    summary["governance_status"] = (
        "Review trigger"
        if summary["governance_status"] == "Review required"
        else "No review trigger"
    )
    return {"summary": summary, "details": fairness}


def disparity_range(values: pd.Series) -> float:
    return values.max() - values.min() if len(values) else 0


def explainability_confidence(model_type: str) -> str:
    return get_model_profile(model_type)["explainability_confidence"]


def explainability_risk(model_type: str) -> str:
    return get_model_profile(model_type)["explainability_risk"]


def governance_score(
    performance: dict[str, float],
    approval_disparity: float,
    status: str,
    model_type: str,
) -> int:
    score = 100
    score -= max(0, int((0.80 - performance["ROC-AUC"]) * 100))
    score -= int(approval_disparity * 120)
    if "High" in status:
        score -= 25
    elif "Review" in status or "Medium" in status:
        score -= 12
    complexity = get_model_profile(model_type)["complexity_level"]
    if complexity == "Medium":
        score -= 8
    if complexity == "High":
        score -= 14
    return max(min(score, 100), 0)


def validation_status(performance: dict[str, float]) -> str:
    if performance["ROC-AUC"] >= 0.75 and performance["KS"] >= 0.30:
        return "Pass"
    if performance["ROC-AUC"] >= 0.65 and performance["KS"] >= 0.20:
        return "Review required"
    return "Fail"


def drift_detection_status(
    reference_data: pd.DataFrame,
    scored_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    """Prototype extension: compare train/test feature means as a simple drift proxy."""
    drift_rows = []
    numeric_features = [
        column
        for column in feature_columns
        if column in reference_data.columns
        and column in scored_data.columns
        and pd.api.types.is_numeric_dtype(reference_data[column])
    ]

    for feature in numeric_features:
        reference_mean = reference_data[feature].mean()
        scored_mean = scored_data[feature].mean()
        reference_std = reference_data[feature].std() or 1
        drift_score = abs(scored_mean - reference_mean) / reference_std
        drift_rows.append(
            {
                "feature": feature,
                "reference_mean": reference_mean,
                "scored_mean": scored_mean,
                "drift_score": drift_score,
            }
        )

    drift_table = pd.DataFrame(drift_rows).sort_values(
        "drift_score", ascending=False
    )
    max_drift = float(drift_table["drift_score"].max()) if len(drift_table) else 0.0
    if max_drift >= get_threshold("drift_warning_score"):
        status = "Prototype drift warning"
    elif max_drift >= get_threshold("drift_monitor_score"):
        status = "Prototype drift monitor"
    else:
        status = "No material drift signal"

    return {
        "status": status,
        "max_drift_score": max_drift,
        "table": drift_table,
    }


def model_risk_management_layer(
    *,
    model_type: str,
    performance: dict[str, float],
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    disparity_threshold: float,
    scored: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: list[str],
    fairness_method: str,
) -> dict[str, object]:
    validation = validation_status(performance)
    confidence = explainability_confidence(model_type)
    drift = drift_detection_status(reference_data, scored, feature_columns)
    manual_review_rate = float(scored["human_review_trigger"].mean())

    fairness_status = (
        "Review required"
        if max(approval_disparity, fpr_disparity, fnr_disparity) > disparity_threshold
        else "Pass"
    )
    human_review_status = (
        "Escalate"
        if manual_review_rate >= get_threshold("high_manual_review_rate")
        else "Monitor"
        if manual_review_rate >= get_threshold("monitor_manual_review_rate")
        else "Standard review"
    )

    return {
        "model_validation": validation,
        "explainability_assessment": confidence,
        "fairness_bias_monitoring": fairness_status,
        "drift_detection": drift,
        "human_review_escalation": human_review_status,
        "manual_review_rate": manual_review_rate,
        "fairness_method": fairness_method,
    }
