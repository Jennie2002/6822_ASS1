import pandas as pd

from src.config import get_config_value, get_model_profile, get_threshold
from src.model_risk import explainability_confidence


OWNER_MAPPING = {
    "FAIRNESS_THRESHOLD_BREACH": "Fair Lending / Compliance Committee",
    "FAIRNESS_SEVERE_BREACH": "Fair Lending / Compliance Committee",
    "EXPLAINABILITY_REVIEW_REQUIRED": "Model Risk Committee",
    "EXPLAINABILITY_EVIDENCE_MISSING": "Model Validation Team",
    "EU_HIGH_RISK_AI_APPLIES": "AI Governance Committee",
    "HUMAN_OVERSIGHT_REQUIRED": "Business Owner + Compliance",
    "MODEL_RISK_COMMITTEE_REVIEW_REQUIRED": "Model Risk Committee",
    "DEPLOYMENT_HOLD_RECOMMENDED": "Senior Governance Committee",
    "DRIFT_WARNING": "Model Validation Team",
}

ACTION_MAPPING = {
    "EU_HIGH_RISK_AI_APPLIES": "complete_bias_testing_evidence",
    "FAIRNESS_THRESHOLD_BREACH": "conduct_fairness_investigation",
    "FAIRNESS_SEVERE_BREACH": "conduct_fairness_investigation",
    "EXPLAINABILITY_REVIEW_REQUIRED": "complete_explainability_review",
    "EXPLAINABILITY_EVIDENCE_MISSING": "generate_shap_lime_evidence",
    "HUMAN_OVERSIGHT_REQUIRED": "document_human_oversight_process",
    "MODEL_RISK_COMMITTEE_REVIEW_REQUIRED": "obtain_model_risk_committee_signoff",
    "DEPLOYMENT_HOLD_RECOMMENDED": "perform_jurisdiction_reassessment",
    "DRIFT_WARNING": "monitor_post_deployment_drift",
}


def explainability_evidence_assessment(
    model_type: str,
    jurisdiction: str,
    shap_available: bool = False,
    lime_available: bool = False,
) -> dict[str, str]:
    """Assess explanation evidence for governance review.

    SHAP/LIME are treated as governance evidence. They support committee review
    and do not automatically approve model deployment.
    """
    profile = get_model_profile(model_type)
    complexity = str(profile["complexity_level"])
    burden = {"Low": "Low", "Medium": "Medium", "High": "High"}[complexity]

    if model_type == "Logistic Regression":
        shap_status = "Not applicable"
        lime_status = "Not applicable"
        evidence_status = "Available"
    elif model_type == "XGBoost":
        shap_status = "Available" if shap_available else "Missing"
        lime_status = "Available" if lime_available else "Optional / not generated"
        evidence_status = "Available" if shap_available or lime_available else "Limited"
    else:
        shap_status = "Available" if shap_available else "Missing"
        lime_status = "Available" if lime_available else "Missing"
        evidence_status = "Limited" if shap_available or lime_available else "Missing"

    review_note = (
        "Native model transparency supports governance review."
        if complexity == "Low"
        else "Post-hoc SHAP/LIME evidence is expected before committee sign-off."
    )
    if jurisdiction == "EU" and complexity in {"Medium", "High"}:
        review_note = (
            "EU high-risk credit governance requires enhanced explainability evidence."
        )

    return {
        "model_complexity": complexity,
        "explainability_burden": burden,
        "shap_evidence_status": shap_status,
        "lime_evidence_status": lime_status,
        "explanation_evidence_status": evidence_status,
        "review_note": review_note,
    }


def add_flag(
    flags: list[dict[str, str]], flag: str, severity: str, message: str
) -> None:
    if not any(existing["flag"] == flag for existing in flags):
        flags.append({"flag": flag, "severity": severity, "message": message})


def required_actions_from_flags(flags: list[dict[str, str]]) -> list[str]:
    actions = [
        ACTION_MAPPING[flag["flag"]]
        for flag in flags
        if flag["flag"] in ACTION_MAPPING
    ]
    actions.append("monitor_post_deployment_drift")
    return sorted(set(actions))


def escalation_owners_from_flags(flags: list[dict[str, str]]) -> list[str]:
    owners = [
        OWNER_MAPPING[flag["flag"]]
        for flag in flags
        if flag["flag"] in OWNER_MAPPING
    ]
    return sorted(set(owners)) or ["None"]


def determine_governance_posture(
    flags: list[dict[str, str]],
    *,
    mrm_validation: str,
    fairness_governance: str,
    model_complexity: str,
    explainability_burden: str,
    explanation_evidence_status: str,
    jurisdiction: str,
) -> str:
    flag_names = {flag["flag"] for flag in flags}

    if "FAIRNESS_SEVERE_BREACH" in flag_names or mrm_validation == "Fail":
        return "Deployment hold recommended"
    if (
        jurisdiction == "EU"
        and model_complexity == "High"
        and explanation_evidence_status == "Missing"
    ):
        return "Deployment hold recommended"
    if (
        jurisdiction == "EU"
        and "FAIRNESS_THRESHOLD_BREACH" in flag_names
        and model_complexity in {"Medium", "High"}
        and explanation_evidence_status == "Missing"
    ):
        return "Deployment hold recommended"
    if fairness_governance == "Review required":
        return "Enhanced review required"
    if model_complexity == "High" and explainability_burden == "High":
        return "Enhanced review required"
    if (
        jurisdiction == "EU"
        and model_complexity in {"Medium", "High"}
        and explanation_evidence_status in {"Missing", "Limited"}
    ):
        return "Enhanced review required"
    if mrm_validation == "Pass":
        return "Deployable with controls"
    return "Enhanced review required"


def governance_summary_text(jurisdiction: str, posture: str) -> str:
    if jurisdiction == "EU":
        return (
            "Under the EU governance profile, this model is eligible for governance "
            "committee consideration only after required high-risk AI controls, "
            "explainability evidence, and human oversight documentation are completed. "
            f"Current governance posture: {posture}."
        )
    return (
        "Under the US governance profile, this model requires risk-based monitoring "
        "and adverse-action explanation support. Final deployment judgement remains "
        f"with human governance committees. Current governance posture: {posture}."
    )


def governance_action_layer(
    *,
    jurisdiction: str,
    model_type: str,
    performance: dict[str, float],
    mrm_validation: str,
    fairness_governance: str,
    shap_available: bool = False,
    lime_available: bool = False,
    approval_disparity: float,
    fpr_disparity: float,
    fnr_disparity: float,
    disparity_threshold: float,
    scored: pd.DataFrame,
    fairness_method: str,
    jurisdiction_status: str,
) -> dict[str, object]:
    """Translate model findings into governance workflow actions."""
    findings = []
    actions = []
    deployment_flags = []
    structured_flags = []

    model_profile = get_model_profile(model_type)
    confidence = explainability_confidence(model_type)
    evidence = explainability_evidence_assessment(
        model_type,
        jurisdiction,
        shap_available=shap_available,
        lime_available=lime_available,
    )
    deployment_gate = model_profile["deployment_gate"]
    severe_threshold = disparity_threshold * get_threshold("severe_disparity_multiplier")
    manual_review_rate = float(scored["human_review_trigger"].mean())
    rejection_rate = float((scored["decision_recommendation"] == "Reject").mean())
    high_risk_rejection_rate = float(
        (
            (scored["decision_recommendation"] == "Reject")
            & (scored["default_probability"] >= 0.35)
        ).mean()
    )

    if jurisdiction == "EU":
        add_flag(
            structured_flags,
            "EU_HIGH_RISK_AI_APPLIES",
            "high",
            "Credit scoring is treated as high-risk AI under the EU governance profile.",
        )
        add_flag(
            structured_flags,
            "HUMAN_OVERSIGHT_REQUIRED",
            "medium",
            "Human oversight is required before adverse credit decisions can be operationalised.",
        )

    if "BISG" in fairness_method and approval_disparity > disparity_threshold:
        findings.append("proxy discrimination risk")
        actions.append(str(get_config_value("severe_bias_action")))
        actions.append("fair-lending review using BISG weighted audit results")

    if approval_disparity > severe_threshold:
        findings.append("severe approval disparity")
        actions.append(str(get_config_value("high_risk_rejection_action")))
        actions.append(str(get_config_value("bias_warning_action")))
        actions.append(str(get_config_value("severe_bias_action")))
        deployment_flags.append("deployment restriction until disparity review is completed")
        add_flag(
            structured_flags,
            "FAIRNESS_SEVERE_BREACH",
            "high",
            "Approval disparity materially exceeds the severe fairness governance threshold.",
        )
    elif approval_disparity > disparity_threshold:
        findings.append("approval disparity above threshold")
        actions.append("fairness remediation review")

    if fpr_disparity > disparity_threshold or fnr_disparity > disparity_threshold:
        findings.append("uneven error distribution across groups")
        actions.append("error-rate bias investigation")

    if (
        approval_disparity > disparity_threshold
        or fpr_disparity > disparity_threshold
        or fnr_disparity > disparity_threshold
    ):
        add_flag(
            structured_flags,
            "FAIRNESS_THRESHOLD_BREACH",
            "high",
            "One or more fairness disparity metrics exceed the selected governance threshold.",
        )

    if manual_review_rate > get_threshold("high_manual_review_rate"):
        findings.append("high manual review rate")
        actions.append("review decision threshold and model calibration")
    elif manual_review_rate > get_threshold("monitor_manual_review_rate"):
        findings.append("elevated manual review rate")
        actions.append("monitor manual review workload")

    if confidence == "Low":
        findings.append("low explainability confidence")
        actions.append(str(get_config_value("low_explainability_action")))
        if jurisdiction == "EU":
            deployment_flags.append("EU deployment requires explainability evidence review")

    if jurisdiction == "EU" and evidence["model_complexity"] in {"Medium", "High"}:
        add_flag(
            structured_flags,
            "EXPLAINABILITY_REVIEW_REQUIRED",
            "medium",
            "Model complexity requires additional explainability evidence under the selected governance profile.",
        )

    if evidence["explanation_evidence_status"] == "Missing":
        add_flag(
            structured_flags,
            "EXPLAINABILITY_EVIDENCE_MISSING",
            "high" if jurisdiction == "EU" else "medium",
            "SHAP/LIME or equivalent explanation evidence is missing for a complex model.",
        )

    if jurisdiction == "EU" and (
        approval_disparity > disparity_threshold
        or fpr_disparity > disparity_threshold
        or fnr_disparity > disparity_threshold
    ):
        deployment_flags.append(
            "EU deployment requires completion of high-risk AI bias and governance review"
        )

    if (
        rejection_rate > get_threshold("high_rejection_rate")
        or high_risk_rejection_rate > get_threshold("high_risk_rejection_rate")
    ):
        findings.append("high-risk automated rejection")
        actions.append(str(get_config_value("high_risk_rejection_action")))
        actions.append("manual review queue for rejected or borderline applicants")
        add_flag(
            structured_flags,
            "HUMAN_OVERSIGHT_REQUIRED",
            "medium",
            "Human oversight is required before adverse credit decisions can be operationalised.",
        )

    if "Restricted" in deployment_gate or "Conditional" in deployment_gate:
        deployment_flags.append(f"model deployment gate: {deployment_gate}")
        add_flag(
            structured_flags,
            "MODEL_RISK_COMMITTEE_REVIEW_REQUIRED",
            "medium",
            "Model governance findings require committee-level review before deployment.",
        )

    if "High risk" in jurisdiction_status:
        if jurisdiction == "EU":
            actions.append("AI governance committee sign-off required")
        else:
            actions.append("senior governance sign-off required")
        deployment_flags.append("jurisdiction high-risk controls require committee review")

    if mrm_validation == "Fail":
        add_flag(
            structured_flags,
            "MODEL_RISK_COMMITTEE_REVIEW_REQUIRED",
            "medium",
            "Model validation weakness requires committee-level model risk review.",
        )

    if not findings:
        findings.append("no critical governance trigger")
        actions.append("continue monitoring")

    posture = determine_governance_posture(
        structured_flags,
        mrm_validation=mrm_validation,
        fairness_governance=fairness_governance,
        model_complexity=evidence["model_complexity"],
        explainability_burden=evidence["explainability_burden"],
        explanation_evidence_status=evidence["explanation_evidence_status"],
        jurisdiction=jurisdiction,
    )
    if posture == "Deployment hold recommended":
        add_flag(
            structured_flags,
            "DEPLOYMENT_HOLD_RECOMMENDED",
            "high",
            "Deployment should remain paused pending governance remediation.",
        )
        deployment_flags.append("deployment hold until governance actions are closed")

    mapped_actions = required_actions_from_flags(structured_flags)
    mapped_owners = escalation_owners_from_flags(structured_flags)
    actions = sorted(set(actions + mapped_actions))
    escalation_status = (
        "Committee review required"
        if mapped_owners != ["None"] or any(flag["severity"] == "high" for flag in structured_flags)
        else "None"
    )
    human_oversight = (
        "Required"
        if jurisdiction == "EU"
        else "Enhanced"
        if (
            fairness_governance == "Review required"
            or evidence["explainability_burden"] == "High"
            or any(flag["flag"] == "DEPLOYMENT_HOLD_RECOMMENDED" for flag in structured_flags)
        )
        else "Risk-based"
    )

    return {
        "findings": sorted(set(findings)),
        "actions": sorted(set(actions)),
        "deployment_flags": sorted(set(deployment_flags))
        or ["no deployment restriction triggered"],
        "explainability_confidence": confidence,
        "manual_review_rate": manual_review_rate,
        "rejection_rate": rejection_rate,
        "high_risk_rejection_rate": high_risk_rejection_rate,
        "deployment_gate": deployment_gate,
        "governance_flags": structured_flags,
        "required_actions": actions,
        "escalation_owners": mapped_owners,
        "escalation_status": escalation_status,
        "governance_posture": posture,
        "governance_summary": governance_summary_text(jurisdiction, posture),
        "human_oversight": human_oversight,
        "explainability_evidence": evidence,
        "mrm_validation": mrm_validation,
        "fairness_governance": fairness_governance,
    }
