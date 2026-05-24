from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MODEL_TYPES, get_model_profile
from src.governance.action_layer import governance_action_layer
from src.jurisdiction import evaluate_jurisdiction_detail, get_jurisdiction_config
from src.model_risk import (
    add_demo_bisg_probabilities,
    compute_bisg_governance_audit,
    compute_multilayer_fairness,
    compute_performance,
    explainability_risk,
    governance_score,
    model_risk_management_layer,
)
from src.model_adapter.credit_model import (
    add_predictions,
    feature_importance,
    load_registered_model,
)
from src.model_adapter.explainability import compute_shap_local_explanation
from src.model_adapter.scorecard import add_scorecard_outputs


def detect_explainability_artifacts(
    *,
    model_type: str,
    model: object,
    X_train: pd.DataFrame,
    scored: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    """Detect whether SHAP/LIME evidence is actually available for governance review.

    Native Logistic Regression interpretability is treated separately from SHAP/LIME.
    SHAP/LIME evidence supports governance review; it is not an automatic approval
    mechanism.
    """
    result = {
        "shap_available": False,
        "lime_available": False,
        "evidence_note": "No SHAP/LIME evidence generated.",
    }
    if model_type == "Logistic Regression":
        result["evidence_note"] = "Native coefficient interpretability is available."
        return result
    if model_type != "XGBoost":
        result["evidence_note"] = (
            "No validated SHAP/LIME evidence artifact is available for this prototype model."
        )
        return result

    shap_result, shap_error = compute_shap_local_explanation(
        model,
        X_train,
        scored.iloc[0],
        feature_columns,
        max_background_rows=10,
    )
    if shap_result is not None and not shap_result.empty:
        result["shap_available"] = True
        result["evidence_note"] = "SHAP evidence generated for governance review."
    else:
        result["evidence_note"] = shap_error or "SHAP evidence could not be generated."
    return result


def assess_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    jurisdiction: str,
    model_type: str,
    decision_threshold: float,
    disparity_threshold: float,
    seed: int,
    bisg_available: bool = False,
) -> dict[str, object]:
    train_data, test_data = train_test_split(
        data,
        test_size=0.25,
        random_state=seed,
        stratify=data["repaid"],
    )
    model = load_registered_model(model_type)
    X_train = train_data[feature_columns]
    y_test = test_data["repaid"]
    scored = add_predictions(
        model, test_data.copy(), feature_columns, decision_threshold
    )
    scored = add_scorecard_outputs(scored)
    performance = compute_performance(
        y_test, scored["approved"], scored["approval_probability"]
    )
    fairness_group_columns = [
        "age_group",
        "income_group",
        "synthetic_protected_group",
    ]
    if jurisdiction == "US" and bisg_available:
        scored = add_demo_bisg_probabilities(scored, seed)
    multilayer_fairness = compute_multilayer_fairness(
        scored,
        fairness_group_columns,
        disparity_threshold,
    )
    bisg_audit = None
    if jurisdiction == "US" and bisg_available:
        bisg_audit = compute_bisg_governance_audit(scored, disparity_threshold)
        proxy_summary = bisg_audit["summary"]
        approval_disparity = max(
            multilayer_fairness["worst_approval_disparity"],
            float(proxy_summary["approval_disparity"]),
        )
        fpr_disparity = max(
            multilayer_fairness["worst_fpr_disparity"],
            float(proxy_summary["fpr_disparity"]),
        )
        fnr_disparity = max(
            multilayer_fairness["worst_fnr_disparity"],
            float(proxy_summary["fnr_disparity"]),
        )
        fairness_method = (
            "Explicit fairness-group audits plus BISG-weighted supplementary proxy monitoring"
        )
    else:
        approval_disparity = multilayer_fairness["worst_approval_disparity"]
        fpr_disparity = multilayer_fairness["worst_fpr_disparity"]
        fnr_disparity = multilayer_fairness["worst_fnr_disparity"]
        fairness_method = "Explicit fairness-group audits without BISG proxy inference"
    selected_fairness = multilayer_fairness["details"].get(group_column)
    if selected_fairness is None and multilayer_fairness["details"]:
        selected_fairness = next(iter(multilayer_fairness["details"].values()))
    if selected_fairness is None:
        selected_fairness = pd.DataFrame()
    jurisdiction_result = evaluate_jurisdiction_detail(
        jurisdiction,
        approval_disparity,
        fpr_disparity,
        fnr_disparity,
        disparity_threshold,
        has_explanation=True,
        bisg_available=bisg_available,
    )
    explanation = feature_importance(model, feature_columns, model_type)
    model_profile = get_model_profile(model_type)
    status = jurisdiction_result["status"]
    # Supporting monitoring indicator only. Governance posture is determined by
    # flags/actions in governance_action_layer, not by this score.
    score = governance_score(performance, approval_disparity, status, model_type)
    model_risk = model_risk_management_layer(
        model_type=model_type,
        performance=performance,
        approval_disparity=approval_disparity,
        fpr_disparity=fpr_disparity,
        fnr_disparity=fnr_disparity,
        disparity_threshold=disparity_threshold,
        scored=scored,
        reference_data=X_train,
        feature_columns=feature_columns,
        fairness_method=fairness_method,
    )
    explanation_artifacts = detect_explainability_artifacts(
        model_type=model_type,
        model=model,
        X_train=X_train,
        scored=scored,
        feature_columns=feature_columns,
    )
    governance_actions = governance_action_layer(
        jurisdiction=jurisdiction,
        model_type=model_type,
        performance=performance,
        mrm_validation=model_risk["model_validation"],
        fairness_governance=model_risk["fairness_bias_monitoring"],
        shap_available=bool(explanation_artifacts.get("shap_available", False)),
        lime_available=bool(explanation_artifacts.get("lime_available", False)),
        approval_disparity=approval_disparity,
        fpr_disparity=fpr_disparity,
        fnr_disparity=fnr_disparity,
        disparity_threshold=disparity_threshold,
        scored=scored,
        fairness_method=fairness_method,
        jurisdiction_status=status,
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": test_data[feature_columns],
        "y_test": y_test,
        "scored": scored,
        "jurisdiction": jurisdiction,
        "performance": performance,
        "fairness": selected_fairness,
        "multilayer_fairness": multilayer_fairness,
        "bisg_audit": bisg_audit,
        "fairness_method": fairness_method,
        "approval_disparity": approval_disparity,
        "fpr_disparity": fpr_disparity,
        "fnr_disparity": fnr_disparity,
        "disparity_threshold": disparity_threshold,
        "status": jurisdiction_result["status"],
        "rule": jurisdiction_result["rule"],
        "flags": jurisdiction_result["flags"],
        "required_controls": jurisdiction_result["required_controls"],
        "bisg_status": jurisdiction_result["bisg_status"],
        "explanation": explanation,
        "model_profile": model_profile,
        "governance_score": score,
        "explainability_risk": explainability_risk(model_type),
        "model_risk": model_risk,
        "explainability_artifacts": explanation_artifacts,
        "governance_actions": governance_actions,
    }


def assess_all_models(
    data: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    jurisdiction: str,
    decision_threshold: float,
    disparity_threshold: float,
    seed: int,
    bisg_available: bool = False,
) -> dict[str, dict[str, object]]:
    return {
        model_type: assess_model(
            data,
            feature_columns,
            group_column,
            jurisdiction,
            model_type,
            decision_threshold,
            disparity_threshold,
            seed,
            bisg_available,
        )
        for model_type in MODEL_TYPES
    }


def assessment_summary_table(
    assessments: dict[str, dict[str, object]]
) -> pd.DataFrame:
    rows = []
    for model_type, assessment in assessments.items():
        performance = assessment["performance"]
        scored = assessment["scored"]
        governance_actions = assessment["governance_actions"]
        model_profile = assessment["model_profile"]
        rows.append(
            {
                "model": model_type,
                "role": model_profile["role"],
                "complexity": model_profile["complexity_level"],
                "deployment_gate": model_profile["deployment_gate"],
                "governance_posture": governance_actions["governance_posture"],
                "mrm_validation": governance_actions["mrm_validation"],
                "fairness_governance": governance_actions["fairness_governance"],
                "escalation_status": governance_actions["escalation_status"],
                "governance_score": assessment["governance_score"],
                "roc_auc": performance["ROC-AUC"],
                "ks": performance["KS"],
                "approval_disparity": assessment["approval_disparity"],
                "fpr_disparity": assessment["fpr_disparity"],
                "fnr_disparity": assessment["fnr_disparity"],
                "breached_fairness_groups": ", ".join(
                    assessment["multilayer_fairness"]["breached_groups"]
                )
                or "None",
                "manual_review_rate": scored["human_review_trigger"].mean(),
                "rejection_rate": governance_actions["rejection_rate"],
                "explainability_risk": assessment["explainability_risk"],
            }
        )
    return pd.DataFrame(rows)


def jurisdiction_context(jurisdiction: str) -> str:
    config = get_jurisdiction_config(jurisdiction)
    return (
        f"{config.name} configuration: {config.report_focus}. "
        f"Rule profile: {config.rule_profile}. "
        f"Rule version: {config.rule_version}. "
        f"Fairness method: {config.fairness_method}. "
        f"BISG policy: {config.bisg_policy}"
    )


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    frame = data.copy()
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def fairness_tests_markdown(assessment: dict[str, object]) -> str:
    multilayer = assessment["multilayer_fairness"]
    summary = multilayer["summary"].copy()
    if not summary.empty:
        for column in ["approval_disparity", "fpr_disparity", "fnr_disparity"]:
            summary[column] = summary[column].astype(float).round(3)

    sections = ["### Explicit Fairness-Group Audits", dataframe_to_markdown(summary)]
    for group_column, table in multilayer["details"].items():
        display = table.copy()
        for column in ["approval_rate", "false_positive_rate", "false_negative_rate"]:
            display[column] = display[column].astype(float).round(3)
        sections.extend(
            [
                f"#### {group_column} fairness audit",
                dataframe_to_markdown(display),
            ]
        )

    bisg_audit = assessment.get("bisg_audit")
    if bisg_audit:
        proxy_summary = pd.DataFrame([bisg_audit["summary"]])
        for column in ["approval_disparity", "fpr_disparity", "fnr_disparity"]:
            proxy_summary[column] = proxy_summary[column].astype(float).round(3)
        proxy_details = bisg_audit["details"].copy()
        for column in ["approval_rate", "false_positive_rate", "false_negative_rate"]:
            proxy_details[column] = proxy_details[column].astype(float).round(3)
        sections.extend(
            [
                "### Additional US Proxy Monitoring",
                "BISG-weighted monitoring is used only for aggregate governance review and fairness-risk assessment. It is not used for individual credit decisions.",
                dataframe_to_markdown(proxy_summary),
                "#### BISG-weighted governance audit",
                dataframe_to_markdown(proxy_details),
            ]
        )

    return "\n\n".join(sections)


def fairness_governance_interpretation(jurisdiction: str, bisg_enabled: bool) -> str:
    if jurisdiction == "US" and bisg_enabled:
        return (
            "The US governance profile supplements explicit fairness-group monitoring "
            "with BISG-weighted proxy disparate-impact monitoring."
        )
    if jurisdiction == "US":
        return (
            "The US governance profile applies explicit fairness-group monitoring. "
            "BISG supplementary proxy monitoring is disabled for this run."
        )
    return (
        "The EU governance profile applies explicit fairness-group monitoring without "
        "BISG-based proxy demographic inference."
    )


def generate_governance_report(
    *,
    data_source: str,
    jurisdiction: str,
    model_type: str,
    group_column: str,
    decision_threshold: float,
    disparity_threshold: float,
    seed: int,
    assessment: dict[str, object],
) -> str:
    performance = assessment["performance"]
    fairness = assessment["fairness"]
    explanation = assessment["explanation"]
    scored = assessment["scored"]
    model_profile = assessment["model_profile"]
    governance_actions = assessment["governance_actions"]
    model_risk = assessment["model_risk"]
    evidence = governance_actions["explainability_evidence"]

    fairness_markdown = dataframe_to_markdown(fairness.round(3))
    fairness_tests = fairness_tests_markdown(assessment)
    fairness_interpretation = fairness_governance_interpretation(
        jurisdiction, assessment.get("bisg_audit") is not None
    )
    explanation_markdown = dataframe_to_markdown(explanation.head(8).round(4))
    flags_markdown = "\n".join(
        f"- **{flag['flag']}** ({flag['severity']}): {flag['message']}"
        for flag in governance_actions["governance_flags"]
    ) or "- No structured governance flags triggered"
    actions_markdown = "\n".join(
        f"- {action}" for action in governance_actions["required_actions"]
    )
    owners_markdown = "\n".join(
        f"- {owner}" for owner in governance_actions["escalation_owners"]
    )
    controls_markdown = "\n".join(
        f"- {control}" for control in assessment["required_controls"]
    )
    drift_markdown = dataframe_to_markdown(
        model_risk["drift_detection"]["table"].head(5).round(3)
    )
    manual_review_rate = scored["human_review_trigger"].mean()

    return f"""# HSBC US/EU Credit Model Governance Assessment Report

Generated date: {date.today().isoformat()}

## 1. Report Scope

This report assesses a credit scoring model under the selected jurisdiction-specific governance logic. It does not select the statistically strongest model. It identifies whether the model is governable under the selected jurisdiction profile.

| Item | Selection |
|---|---|
| Data source | {data_source} |
| Jurisdiction | {jurisdiction} |
| Model under review | {model_type} |
| Model role | {model_profile["role"]} |
| Model lifecycle status | {model_profile["lifecycle_status"]} |
| Model complexity | {model_profile["complexity_level"]} |
| Deployment gate | {model_profile["deployment_gate"]} |
| Explicit fairness groups | age_group, income_group, synthetic_protected_group |
| Approval threshold | {decision_threshold:.2f} |
| Disparity threshold | {disparity_threshold:.2f} |
| Random seed | {seed} |

## 2. Governance Outcome

| Indicator | Result |
|---|---|
| Governance posture | {governance_actions["governance_posture"]} |
| Governance summary | {governance_actions["governance_summary"]} |
| Escalation status | {governance_actions["escalation_status"]} |
| Human oversight | {governance_actions["human_oversight"]} |
| Governance monitoring indicator | {assessment["governance_score"]}/100 |

Final deployment judgement remains with HSBC governance committees.

## 3. Governance Summary Cards

| Governance Area | Assessment |
|---|---|
| MRM validation | {governance_actions["mrm_validation"]} |
| Fairness governance | {governance_actions["fairness_governance"]} |
| Explainability burden | {evidence["explainability_burden"]} |
| Explanation evidence | {evidence["explanation_evidence_status"]} |
| Human oversight | {governance_actions["human_oversight"]} |
| Escalation status | {governance_actions["escalation_status"]} |

## 4. Jurisdiction Context

{jurisdiction_context(jurisdiction)}

Rule engine summary:

{assessment["rule"]}

Required controls:

{controls_markdown}

## 5. MRM Validation

| MRM Control | Output |
|---|---|
| Model validation | {model_risk["model_validation"]} |
| Drift detection | {model_risk["drift_detection"]["status"]} |
| Human review escalation | {model_risk["human_review_escalation"]} |
| Manual review rate | {manual_review_rate:.1%} |
| Automated rejection rate | {governance_actions["rejection_rate"]:.1%} |

Model performance metrics are MRM validation evidence. They do not automatically determine deployment.

## 6. Fairness Governance

Fairness method: {assessment["fairness_method"]}

| Fairness Metric | Value |
|---|---:|
| Approval rate disparity | {assessment["approval_disparity"]:.3f} |
| False positive rate disparity | {assessment["fpr_disparity"]:.3f} |
| False negative rate disparity | {assessment["fnr_disparity"]:.3f} |
| Selected disparity threshold | {disparity_threshold:.3f} |
| Fairness governance judgement | {governance_actions["fairness_governance"]} |

Governance interpretation:

{fairness_interpretation}

{fairness_tests}

## 7. Explainability Evidence

| Evidence Item | Status |
|---|---|
| Model complexity | {evidence["model_complexity"]} |
| Explainability burden | {evidence["explainability_burden"]} |
| SHAP evidence | {evidence["shap_evidence_status"]} |
| LIME evidence | {evidence["lime_evidence_status"]} |
| Explanation evidence | {evidence["explanation_evidence_status"]} |
| Explainability confidence | {governance_actions["explainability_confidence"]} |

Explanation note:

Feature importance, SHAP, and LIME are explainability evidence for governance review. They are not automatic deployment mechanisms and are not complete legal explanations for individual credit decisions.

## 8. Governance Flags

{flags_markdown}

## 9. Required Governance Actions

{actions_markdown}

## 10. Escalation Owners

{owners_markdown}

## 11. Human Committee Review Note

Final deployment judgement remains with HSBC governance committees, compliance, legal, and senior management. This report provides governance assessment support only.

## 12. Technical Appendix

### MRM Performance Evidence

| Metric | Value |
|---|---:|
| Accuracy | {performance["Accuracy"]:.3f} |
| Precision | {performance["Precision"]:.3f} |
| Recall | {performance["Recall"]:.3f} |
| ROC-AUC | {performance["ROC-AUC"]:.3f} |
| KS | {performance["KS"]:.3f} |

### Credit Scorecard Summary

| Scorecard Item | Value |
|---|---:|
| Average credit score | {scored["credit_score"].mean():.0f} |
| Median default probability | {scored["default_probability"].median():.3f} |
| Approval rate | {scored["approved"].mean():.1%} |
| Manual review trigger rate | {manual_review_rate:.1%} |

### Fairness by Group

{fairness_markdown}

### Top Model Drivers

{explanation_markdown}

### Prototype Drift Signals

Drift detection is a prototype extension. It compares reference and scored feature distributions as an early warning signal, not as a production drift test.

{drift_markdown}

## 13. Limitations

- This is a university prototype, not legal advice.
- The rule engine is simplified.
- The data is harmonised from benchmark datasets with generated governance support fields.
- The scorecard is not calibrated on a real bank portfolio.
- The model does not produce production-ready adverse action notices.
- A real deployment would require audit logs, legal review, model validation, and monitoring.

## 14. What This Report Does Not Do

- It does not replace legal advice.
- It does not determine the legally correct jurisdiction automatically.
- It does not prove absence of discrimination.
- It does not retrain or remediate the model automatically.
- It does not generate production-ready adverse action notices.
"""


def generate_multi_model_governance_report(
    *,
    data_source: str,
    jurisdiction: str,
    group_column: str,
    decision_threshold: float,
    disparity_threshold: float,
    seed: int,
    assessments: dict[str, dict[str, object]],
) -> str:
    summary = assessment_summary_table(assessments)
    config = get_jurisdiction_config(jurisdiction)
    display_summary = summary.copy()
    display_summary = display_summary.rename(
        columns={"governance_score": "governance_monitoring_indicator"}
    )
    for column in [
        "roc_auc",
        "ks",
        "approval_disparity",
        "fpr_disparity",
        "fnr_disparity",
        "manual_review_rate",
        "rejection_rate",
    ]:
        display_summary[column] = display_summary[column].astype(float).round(3)
    summary_markdown = dataframe_to_markdown(display_summary)

    posture_counts = summary["governance_posture"].value_counts().to_dict()
    posture_summary = ", ".join(
        f"{posture}: {count}" for posture, count in posture_counts.items()
    )
    restricted = summary[
        summary["deployment_gate"].astype(str).str.contains("Restricted", case=False)
    ]
    restricted_models = (
        ", ".join(restricted["model"].tolist()) if len(restricted) else "None"
    )
    us_specific_finding = ""
    if jurisdiction == "US":
        bisg_audits = [
            assessment.get("bisg_audit")
            for assessment in assessments.values()
            if assessment.get("bisg_audit")
        ]
        bisg_triggered = any(
            audit["summary"]["governance_status"] == "Review trigger"
            for audit in bisg_audits
        )
        bisg_status = (
            "BISG monitoring triggered supplementary review in this run."
            if bisg_triggered
            else "BISG monitoring did not trigger additional review in this run."
        )
        us_specific_finding = (
            "Under the US profile, BISG-weighted proxy monitoring is used as "
            "supplementary aggregate fair-lending monitoring, while final governance "
            "posture is still driven by explicit fairness-group breaches, MRM validation, "
            f"and explainability burden. {bisg_status}"
        )
    uniform_posture_finding = ""
    if len(posture_counts) == 1:
        only_posture = next(iter(posture_counts))
        breached_groups = sorted(
            {
                group
                for assessment in assessments.values()
                for group in assessment["multilayer_fairness"]["breached_groups"]
            }
        )
        if only_posture == "Enhanced review required" and breached_groups:
            uniform_posture_finding = (
                "All models require enhanced review because at least one explicit "
                "fairness-group audit exceeded the selected disparity threshold. "
                f"Breached fairness groups observed: {', '.join(breached_groups)}."
            )

    model_sections = []
    for model_type, assessment in assessments.items():
        performance = assessment["performance"]
        governance_actions = assessment["governance_actions"]
        model_risk = assessment["model_risk"]
        model_profile = assessment["model_profile"]
        evidence = governance_actions["explainability_evidence"]
        fairness_tests = fairness_tests_markdown(assessment)
        fairness_interpretation = fairness_governance_interpretation(
            jurisdiction, assessment.get("bisg_audit") is not None
        )
        structured_flags_markdown = "\n".join(
            f"- **{flag['flag']}** ({flag['severity']}): {flag['message']}"
            for flag in governance_actions["governance_flags"]
        ) or "- No structured governance flags triggered"
        actions_markdown = "\n".join(
            f"- {action}" for action in governance_actions["required_actions"]
        )
        owners_markdown = "\n".join(
            f"- {owner}" for owner in governance_actions["escalation_owners"]
        )
        model_sections.append(
            f"""## Model Review: {model_type}

| Item | Result |
|---|---|
| Role | {model_profile["role"]} |
| Complexity | {model_profile["complexity_level"]} |
| Deployment gate | {model_profile["deployment_gate"]} |
| Governance posture | {governance_actions["governance_posture"]} |
| Escalation status | {governance_actions["escalation_status"]} |
| Human oversight | {governance_actions["human_oversight"]} |
| Governance monitoring indicator | {assessment["governance_score"]}/100 |
| ROC-AUC | {performance["ROC-AUC"]:.3f} |
| KS | {performance["KS"]:.3f} |
| Approval disparity | {assessment["approval_disparity"]:.3f} |
| FPR disparity | {assessment["fpr_disparity"]:.3f} |
| FNR disparity | {assessment["fnr_disparity"]:.3f} |
| Breached fairness groups | {", ".join(assessment["multilayer_fairness"]["breached_groups"]) or "None"} |
| Explainability risk | {assessment["explainability_risk"]} |
| Model validation | {model_risk["model_validation"]} |
| Fairness / bias monitoring | {model_risk["fairness_bias_monitoring"]} |
| Human review escalation | {model_risk["human_review_escalation"]} |
| SHAP evidence | {evidence["shap_evidence_status"]} |
| LIME evidence | {evidence["lime_evidence_status"]} |
| Explanation evidence | {evidence["explanation_evidence_status"]} |

Governance summary:

{governance_actions["governance_summary"]}

Governance flags:

{structured_flags_markdown}

Required governance actions:

{actions_markdown}

Escalation owner:

{owners_markdown}

Human committee review note:

Final deployment judgement remains with human governance committees.

Fairness governance interpretation:

{fairness_interpretation}

{fairness_tests}
"""
        )

    model_sections_markdown = "\n".join(model_sections)

    return f"""# HSBC US/EU Credit Model Governance Assessment Report

Generated date: {date.today().isoformat()}

## 1. Report Scope

This report evaluates all candidate credit scoring model artefacts under the selected jurisdiction. The jurisdiction and governance settings are supplied at runtime; the tool assesses Logistic Regression, XGBoost, and FNN in sequence.

| Item | Selection |
|---|---|
| Data source | {data_source} |
| Jurisdiction | {jurisdiction} |
| Rule profile | {config.rule_profile} |
| Rule version | {config.rule_version} |
| Assessment date | {date.today().isoformat()} |
| Models reviewed | {", ".join(assessments.keys())} |
| Explicit fairness groups | age_group, income_group, synthetic_protected_group |
| Approval threshold | {decision_threshold:.2f} |
| Disparity threshold | {disparity_threshold:.2f} |
| Random seed | {seed} |

## 2. Jurisdiction Context

{jurisdiction_context(jurisdiction)}

## 3. Executive Summary

Governance posture distribution: **{posture_summary}**.

Models with restricted deployment gate: **{restricted_models}**.

This report does not select the statistically strongest model. It identifies which model is most governable under the selected jurisdiction profile.

The governance monitoring indicator is supporting context only. The final deployment judgement remains with HSBC governance committees. Accuracy, ROC-AUC, KS, and monitoring scores do not automatically determine deployment.

{us_specific_finding}

{uniform_posture_finding}

## 4. Cross-Model Governance Scorecard

{summary_markdown}

{model_sections_markdown}

## 5. Limitations

- This is a university prototype, not legal advice.
- The data is a harmonised benchmark-based governance dataset, not a real bank portfolio.
- The rule engine is simplified.
- SHAP, LIME, and feature importance support governance review but are not complete legal explanations.
- A real deployment would require audit logs, legal review, independent model validation, monitoring, and documented human oversight.

## 6. What This Tool Does Not Do

- It does not replace legal advice.
- It does not determine the legally correct jurisdiction automatically.
- It does not prove absence of discrimination.
- It does not retrain, remediate, or approve a model automatically.
- It does not generate production-ready adverse action notices.

## 7. Footer Disclaimer

This dashboard provides governance assessment support only. It does not automatically approve model deployment. Final deployment judgement remains with HSBC governance committees, compliance, legal, and senior management.
"""
