from dataclasses import dataclass

from src.config import load_config_file


@dataclass(frozen=True)
class JurisdictionConfig:
    code: str
    name: str
    engine: str
    rule_profile: str
    rule_version: str
    fairness_method: str
    bisg_policy: str
    report_focus: str
    required_controls: list[str]
    high_risk_ai: bool = False
    ai_act_category: str = ""
    adverse_action_required: bool = False
    disparate_impact_review: str = ""


def load_jurisdiction_config(filename: str) -> JurisdictionConfig:
    raw = load_config_file(filename)
    return JurisdictionConfig(
        code=str(raw["code"]),
        name=str(raw["name"]),
        engine=str(raw["engine"]),
        rule_profile=str(raw.get("rule_profile", raw["engine"])),
        rule_version=str(raw.get("rule_version", "")),
        fairness_method=str(raw["fairness_method"]),
        bisg_policy=str(raw["bisg_policy"]),
        report_focus=str(raw["report_focus"]),
        required_controls=list(raw["required_controls"]),
        high_risk_ai=bool(raw.get("high_risk_ai", False)),
        ai_act_category=str(raw.get("ai_act_category", "")),
        adverse_action_required=bool(raw.get("adverse_action_required", False)),
        disparate_impact_review=str(raw.get("disparate_impact_review", "")),
    )


JURISDICTION_CONFIGS = {
    "US": load_jurisdiction_config("us_config.yaml"),
    "EU": load_jurisdiction_config("eu_config.yaml"),
}

JURISDICTIONS = list(JURISDICTION_CONFIGS.keys())


def get_jurisdiction_config(jurisdiction: str) -> JurisdictionConfig:
    if jurisdiction not in JURISDICTION_CONFIGS:
        raise ValueError(f"Unsupported jurisdiction: {jurisdiction}")
    return JURISDICTION_CONFIGS[jurisdiction]


def evaluate_jurisdiction(
    jurisdiction: str,
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    threshold: float,
    has_explanation: bool,
    bisg_available: bool = False,
) -> tuple[str, str, list[str]]:
    result = evaluate_jurisdiction_detail(
        jurisdiction,
        approval_disparity,
        fpr_disparity,
        fnr_disparity,
        threshold,
        has_explanation,
        bisg_available,
    )
    return result["status"], result["rule"], result["flags"]


def evaluate_jurisdiction_detail(
    jurisdiction: str,
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    threshold: float,
    has_explanation: bool,
    bisg_available: bool = False,
) -> dict[str, object]:
    if jurisdiction == "US":
        return evaluate_us_engine(
            approval_disparity,
            fpr_disparity,
            fnr_disparity,
            threshold,
            has_explanation,
            bisg_available,
        )
    if jurisdiction == "EU":
        return evaluate_eu_engine(
            approval_disparity,
            fpr_disparity,
            fnr_disparity,
            threshold,
            has_explanation,
            bisg_available,
        )
    raise ValueError(f"Unsupported jurisdiction: {jurisdiction}")


def evaluate_us_engine(
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    threshold: float,
    has_explanation: bool,
    bisg_available: bool = False,
) -> dict[str, object]:
    flags = []
    config = get_jurisdiction_config("US")
    required_controls = list(config.required_controls)

    if bisg_available:
        flags.append(
            "BISG-style proxy monitoring is available for aggregate US fair-lending review."
        )
    else:
        flags.append(
            "BISG-style proxy monitoring was not applied because surname/geography data is unavailable."
        )

    if approval_disparity > threshold:
        flags.append(
            "Approval rate disparity exceeds threshold; fair-lending litigation and reputational risk should be reviewed."
        )
    if fpr_disparity > threshold:
        flags.append(
            "False positive rate disparity exceeds threshold; underwriting error distribution should be reviewed."
        )
    if fnr_disparity > threshold:
        flags.append(
            "False negative rate disparity exceeds threshold; missed-good-applicant risk should be reviewed."
        )
    if config.adverse_action_required:
        if has_explanation:
            flags.append(
                "Adverse action explanation control is available for declined or borderline credit decisions."
            )
        else:
            flags.append(
                "Adverse action explanation gap: specific credit denial reasons are unavailable."
            )

    review_required = (
        approval_disparity > threshold
        or fpr_disparity > threshold
        or fnr_disparity > threshold
        or not has_explanation
    )
    status = "Medium risk / review required" if review_required else "Low risk / monitor"
    rule = (
        "US engine: apply fair-lending risk monitoring, use BISG-style proxy analysis only "
        "for aggregate monitoring when data is available, and flag litigation/reputational risk "
        "when disparity exceeds threshold. Disparate impact review is framed as governance "
        "monitoring and litigation-risk review."
    )

    return {
        "status": status,
        "rule": rule,
        "flags": flags,
        "required_controls": required_controls,
        "bisg_status": "available" if bisg_available else "not available",
    }


def evaluate_eu_engine(
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    threshold: float,
    has_explanation: bool,
    bisg_available: bool = False,
) -> dict[str, object]:
    _ = bisg_available
    flags = []
    config = get_jurisdiction_config("EU")
    required_controls = list(config.required_controls)

    flags.append(
        "BISG-style inferred race estimation is not used in the EU configuration."
    )
    if config.high_risk_ai:
        flags.append(
            f"EU high-risk AI classification applies: {config.ai_act_category}."
        )

    if approval_disparity > threshold:
        flags.append("Bias testing required: approval rate disparity exceeds threshold.")
    if fpr_disparity > threshold:
        flags.append("Bias testing required: false positive rate disparity exceeds threshold.")
    if fnr_disparity > threshold:
        flags.append("Bias testing required: false negative rate disparity exceeds threshold.")
    if not has_explanation:
        flags.append("Explainability required: model explanation output is unavailable.")

    breach = (
        approval_disparity > threshold
        or fpr_disparity > threshold
        or fnr_disparity > threshold
        or not has_explanation
    )
    status = "High risk / action required" if breach else "Monitor"
    rule = (
        "EU engine: treat credit scoring as a high-risk AI governance context requiring "
        "bias testing, explainability evidence, audit trail, and human oversight. Do not apply "
        "BISG-style inferred race proxy estimation in this configuration."
    )

    return {
        "status": status,
        "rule": rule,
        "flags": flags,
        "required_controls": required_controls,
        "bisg_status": "not used",
    }
