import argparse
from html import escape

import streamlit as st

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    MODEL_TYPES,
    get_threshold,
)
from src.io.dataset_loader import load_dataset
from src.jurisdiction import get_jurisdiction_config
from src.reporting import (
    assess_all_models,
    assessment_summary_table,
    generate_multi_model_governance_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static governance report.")
    parser.add_argument(
        "--jurisdiction",
        choices=["US", "EU"],
        required=True,
        help="Jurisdiction to assess: US or EU.",
    )
    parser.add_argument(
        "--fairness-group",
        default="age_group",
        help="Fairness group column. Defaults to age_group.",
    )
    parser.add_argument(
        "--approval-threshold",
        type=float,
        default=None,
        help="Approval threshold for binary decisions. Defaults to 0.45 for EU and 0.50 for US.",
    )
    parser.add_argument(
        "--disparity-threshold",
        type=float,
        default=get_threshold("approval_disparity_threshold"),
        help="Fairness disparity threshold.",
    )
    parser.add_argument(
        "--no-bisg",
        action="store_true",
        help="Disable BISG soft probability weighting for US aggregate fairness monitoring.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic holdout evaluation.",
    )
    args, _ = parser.parse_known_args()
    if args.approval_threshold is None:
        args.approval_threshold = (
            0.45 if args.jurisdiction == "EU" else DEFAULT_DECISION_THRESHOLD
        )
    return args


def inject_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stAlert"] {
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .gov-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }
        .gov-card {
            border: 1px solid #d8dbe2;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
            min-height: 92px;
        }
        .gov-card-label {
            font-size: 0.78rem;
            color: #5f6673;
            margin-bottom: 0.35rem;
            line-height: 1.25;
        }
        .gov-card-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #252936;
            line-height: 1.25;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .gov-mini-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 0.8rem 0;
        }
        .gov-mini-card {
            border: 1px solid #e1e4ea;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: #fbfcfe;
        }
        .gov-mini-value {
            font-size: 1rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }
        .gov-section-note {
            color: #5f6673;
            font-size: 0.88rem;
            line-height: 1.4;
        }
        .posture-banner {
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1rem 0;
            border: 1px solid;
        }
        .posture-green {
            background: #eaf7ef;
            border-color: #92cfa6;
            color: #145a2a;
        }
        .posture-amber {
            background: #fff4df;
            border-color: #e3b862;
            color: #7a4b00;
        }
        .posture-red {
            background: #fde8e8;
            border-color: #dd8a8a;
            color: #8a1f1f;
        }
        .posture-title {
            font-size: 1.2rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }
        .flag-card {
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.55rem;
            border: 1px solid #d8dbe2;
            background: #fff;
        }
        .flag-high {
            border-color: #dd8a8a;
            background: #fdecec;
        }
        .flag-medium {
            border-color: #e3b862;
            background: #fff6e6;
        }
        .flag-low {
            border-color: #9db7d7;
            background: #edf4fb;
        }
        .flag-title {
            font-weight: 750;
            margin-bottom: 0.25rem;
            overflow-wrap: anywhere;
        }
        .notice-box {
            border: 1px solid #d8dbe2;
            border-left: 5px solid #6b7280;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #fbfcfe;
            margin: 0.75rem 0 1rem 0;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card_grid(cards: list[tuple[str, str]]) -> None:
    html_cards = []
    for label, value in cards:
        html_cards.append(
            "<div class='gov-card'>"
            f"<div class='gov-card-label'>{escape(str(label))}</div>"
            f"<div class='gov-card-value'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='gov-card-grid'>" + "".join(html_cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_mini_grid(cards: list[tuple[str, str]]) -> None:
    html_cards = []
    for label, value in cards:
        html_cards.append(
            "<div class='gov-mini-card'>"
            f"<div class='gov-card-label'>{escape(str(label))}</div>"
            f"<div class='gov-mini-value'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='gov-mini-grid'>" + "".join(html_cards) + "</div>",
        unsafe_allow_html=True,
    )


def posture_class(posture: str) -> str:
    if posture == "Deployment hold recommended":
        return "posture-red"
    if posture == "Enhanced review required":
        return "posture-amber"
    return "posture-green"


def severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity.lower(), 3)


def format_summary_table(summary):
    display_summary = summary.copy()
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
    display_summary["governance_score"] = display_summary["governance_score"].astype(int)
    display_summary = display_summary.rename(
        columns={"governance_score": "governance_monitoring_indicator"}
    )
    return display_summary


def render_posture_banner(
    jurisdiction: str, model_type: str, assessment: dict[str, object]
) -> None:
    actions = assessment["governance_actions"]
    posture = actions["governance_posture"]
    flags = actions["governance_flags"]
    reason = actions["governance_summary"]
    st.markdown(
        f"""
        <div class="posture-banner {posture_class(posture)}">
            <div class="posture-title">{escape(jurisdiction)} | {escape(model_type)} | Governance posture: {escape(posture)}</div>
            <div>{escape(reason)}</div>
            <br>
            <div><strong>Human governance note:</strong> Final deployment judgement remains with human governance committees.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if flags:
        high_flags = [flag["flag"] for flag in flags if flag["severity"] == "high"]
        if high_flags:
            st.caption("High severity flags: " + ", ".join(high_flags))


def render_summary_cards(assessment: dict[str, object]) -> None:
    actions = assessment["governance_actions"]
    evidence = actions["explainability_evidence"]
    render_card_grid([
        ("MRM validation", actions["mrm_validation"]),
        ("Fairness governance", actions["fairness_governance"]),
        ("Explainability burden", evidence["explainability_burden"]),
        ("Explanation evidence", evidence["explanation_evidence_status"]),
        ("Human oversight", actions["human_oversight"]),
        ("Escalation status", actions["escalation_status"]),
    ])


def render_flags_actions(assessment: dict[str, object]) -> None:
    actions = assessment["governance_actions"]
    st.subheader("Governance Flags, Actions, and Owners")
    flags = actions["governance_flags"]
    if not flags:
        st.success("No structured governance flags triggered")
    else:
        sorted_flags = sorted(flags, key=lambda flag: severity_rank(flag["severity"]))
        for severity in ["high", "medium", "low"]:
            severity_flags = [
                flag for flag in sorted_flags if flag["severity"].lower() == severity
            ]
            if not severity_flags:
                continue
            st.markdown(f"**{severity.title()} Severity Flags**")
            for flag in severity_flags:
                st.markdown(
                    f"""
                    <div class="flag-card flag-{escape(severity)}">
                        <div class="flag-title">{escape(flag["flag"])} ({escape(flag["severity"])})</div>
                        <div>{escape(flag["message"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Required governance actions", expanded=True):
            for action in actions["required_actions"]:
                st.write(f"- {action}")
    with col2:
        with st.expander("Escalation owner mapping", expanded=True):
            for owner in actions["escalation_owners"]:
                st.write(f"- {owner}")


def render_model_section(model_type: str, assessment: dict[str, object]) -> None:
    profile = assessment["model_profile"]
    performance = assessment["performance"]
    actions = assessment["governance_actions"]
    model_risk = assessment["model_risk"]
    evidence = actions["explainability_evidence"]

    st.header(model_type)
    st.caption(
        f"{profile['role']} | Complexity: {profile['complexity_level']} | "
        f"Explainability risk: {profile['explainability_risk']}"
    )

    render_posture_banner(assessment["jurisdiction"], model_type, assessment)
    render_summary_cards(assessment)
    render_flags_actions(assessment)

    st.subheader("MRM Validation Panel")
    render_mini_grid([
        ("Accuracy", f"{performance['Accuracy']:.3f}"),
        ("ROC-AUC", f"{performance['ROC-AUC']:.3f}"),
        ("KS", f"{performance['KS']:.3f}"),
        ("Precision", f"{performance['Precision']:.3f}"),
        ("Recall", f"{performance['Recall']:.3f}"),
        ("Validation status", model_risk["model_validation"]),
    ])
    st.caption(
        "MRM validation metrics are evidence for review. They do not automatically determine deployment."
    )

    st.subheader("Fairness Governance Panel")
    fairness_status = (
        "Review required"
        if actions["fairness_governance"] == "Review required"
        else "Pass"
    )
    render_mini_grid([
        ("Approval disparity", f"{assessment['approval_disparity']:.3f} ({fairness_status})"),
        ("FPR disparity", f"{assessment['fpr_disparity']:.3f} ({fairness_status})"),
        ("FNR disparity", f"{assessment['fnr_disparity']:.3f} ({fairness_status})"),
        ("Selected threshold", f"{assessment['disparity_threshold']:.3f}"),
    ])
    st.caption(f"Fairness method: {assessment['fairness_method']}")

    breached_groups = assessment["multilayer_fairness"]["breached_groups"]
    st.write(
        "- Explicit fairness groups assessed: age_group, income_group, synthetic_protected_group\n"
        f"- Breached fairness groups: {', '.join(breached_groups) if breached_groups else 'None'}"
    )
    if assessment["jurisdiction"] == "EU":
        st.info(
            "The EU governance profile applies explicit fairness-group monitoring without BISG-based proxy demographic inference."
        )
    elif assessment.get("bisg_audit"):
        st.info(
            "Proxy Monitoring Notice: BISG-weighted monitoring is used only for aggregate governance review and fairness-risk assessment. It is not used for individual credit decisions."
        )
    else:
        st.info(
            "The US governance profile applies explicit fairness-group monitoring. BISG supplementary proxy monitoring is disabled for this run."
        )

    fairness_tabs = st.tabs(
        list(assessment["multilayer_fairness"]["details"].keys())
        + (["BISG soft weighting"] if assessment.get("bisg_audit") else [])
    )
    fairness_tables = list(assessment["multilayer_fairness"]["details"].items())
    for tab, (fairness_group, fairness_table) in zip(fairness_tabs, fairness_tables):
        with tab:
            display = fairness_table.copy()
            for column in ["approval_rate", "false_positive_rate", "false_negative_rate"]:
                display[column] = display[column].astype(float).round(3)
            st.dataframe(display, width="stretch", hide_index=True)
    if assessment.get("bisg_audit"):
        with fairness_tabs[-1]:
            proxy_summary = assessment["bisg_audit"]["summary"]
            render_mini_grid([
                ("US BISG proxy monitoring", "Active"),
                ("Supplementary monitoring status", proxy_summary["governance_status"]),
                ("BISG approval disparity", f"{proxy_summary['approval_disparity']:.3f}"),
                ("BISG FPR / FNR disparity", f"{proxy_summary['fpr_disparity']:.3f} / {proxy_summary['fnr_disparity']:.3f}"),
            ])
            display = assessment["bisg_audit"]["details"].copy()
            for column in ["approval_rate", "false_positive_rate", "false_negative_rate"]:
                display[column] = display[column].astype(float).round(3)
            st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Explainability Evidence Panel")
    render_mini_grid([
        ("Model complexity", evidence["model_complexity"]),
        ("Burden", evidence["explainability_burden"]),
        ("SHAP evidence", evidence["shap_evidence_status"]),
        ("LIME evidence", evidence["lime_evidence_status"]),
        ("Evidence status", evidence["explanation_evidence_status"]),
    ])
    st.caption(evidence["review_note"])
    artifact_note = assessment.get("explainability_artifacts", {}).get("evidence_note")
    if artifact_note:
        st.info(artifact_note)
    if evidence["shap_evidence_status"] == "Available":
        st.success("SHAP explanation evidence generated for governance review.")
    if evidence["lime_evidence_status"] == "Available":
        st.success("LIME explanation evidence generated for governance review.")

    st.subheader("Human Oversight Panel")
    st.write(
        f"- Oversight level: {actions['human_oversight']}\n"
        f"- Manual review rate: {actions['manual_review_rate']:.1%}\n"
        f"- Automated rejection rate: {actions['rejection_rate']:.1%}\n"
        "- Human committee review note: Final deployment judgement remains with human governance committees."
    )

    st.subheader("Top Feature Importance")
    explanation = assessment["explanation"].head(8)
    st.bar_chart(explanation.set_index("feature"))
    st.dataframe(explanation, width="stretch")


def main() -> None:
    args = parse_args()
    jurisdiction = args.jurisdiction
    bisg_available = jurisdiction == "US" and not args.no_bisg

    st.set_page_config(
        page_title=f"{jurisdiction} AI Credit Model Governance Report",
        layout="wide",
    )
    inject_dashboard_style()

    st.title("HSBC US/EU Credit Model Governance Dashboard")
    st.caption(
        "Internal governance decision-support portal. The dashboard generates assessment outcomes, "
        "governance flags, required actions, escalation owners, and governance posture."
    )
    st.markdown(
        """
        <div class="notice-box">
            <strong>Governance Difference Notice.</strong>
            The same model may trigger different governance obligations across jurisdictions.
            The EU profile applies stricter governance over explainability and proxy demographic inference,
            while the US profile extends fair lending monitoring through BISG-weighted supplementary monitoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset = load_dataset(seed=args.seed)
    group_column = (
        args.fairness_group
        if args.fairness_group in dataset.fairness_groups
        else dataset.fairness_groups[0]
    )

    assessments = assess_all_models(
        dataset.data,
        dataset.feature_columns,
        group_column,
        jurisdiction,
        args.approval_threshold,
        args.disparity_threshold,
        args.seed,
        bisg_available,
    )
    summary = assessment_summary_table(assessments)
    report = generate_multi_model_governance_report(
        data_source="Combined benchmark governance dataset",
        jurisdiction=jurisdiction,
        group_column=group_column,
        decision_threshold=args.approval_threshold,
        disparity_threshold=args.disparity_threshold,
        seed=args.seed,
        assessments=assessments,
    )

    st.info(dataset.message)
    config = get_jurisdiction_config(jurisdiction)
    st.subheader("Assessment Settings")
    st.write(
        f"- Jurisdiction: {jurisdiction}\n"
        f"- Rule profile: {config.rule_profile}\n"
        f"- Rule version: {config.rule_version}\n"
        f"- Explicit fairness groups: age_group, income_group, synthetic_protected_group\n"
        f"- Approval threshold: {args.approval_threshold:.2f}\n"
        f"- Disparity threshold: {args.disparity_threshold:.2f}\n"
        f"- BISG supplementary proxy monitoring: {'Enabled' if bisg_available else 'Disabled'}"
    )

    st.subheader("Jurisdiction Rule Profile")
    if jurisdiction == "EU":
        rule_profile = {
            "fairness_testing": "mandatory",
            "explainability_review": "required",
            "human_oversight": "mandatory",
            "bisg_policy": "not_used",
            "high_risk_ai_status": "applies",
            "adverse_decision_review": "required",
            "governance_standard": "EU AI Act high-risk credit scoring profile",
        }
    else:
        rule_profile = {
            "fairness_testing": "governance_monitoring",
            "explainability_review": "adverse_action_support",
            "human_oversight": "risk_based",
            "bisg_policy": "permitted_for_monitoring",
            "high_risk_ai_status": "not_applicable",
            "adverse_decision_review": "required",
            "governance_standard": "US fair lending / model risk governance profile",
        }
    st.json(rule_profile)

    st.subheader("Cross-Model Governance Posture Comparison")
    st.dataframe(format_summary_table(summary), width="stretch")

    posture_counts = summary["governance_posture"].value_counts()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Reviewed", len(summary))
    with col2:
        st.metric("Postures Observed", len(posture_counts))
    with col3:
        st.metric("Jurisdiction", jurisdiction)
    st.caption(
        "The governance monitoring indicator is supporting context only. It is not an approval score or automatic model selection."
    )

    st.header("Model Governance Assessments")
    model_tabs = st.tabs(MODEL_TYPES)
    for tab, model_type in zip(model_tabs, MODEL_TYPES):
        with tab:
            render_model_section(model_type, assessments[model_type])

    st.header("Downloadable Governance Report")
    st.download_button(
        "Download Markdown Report",
        report,
        file_name=f"governance_report_{jurisdiction}_all_models.md",
        mime="text/markdown",
    )
    st.text_area("Report preview", report, height=650)
    st.caption(
        "This dashboard provides governance assessment support only. It does not automatically approve model deployment. "
        "Final deployment judgement remains with HSBC governance committees, compliance, legal, and senior management."
    )


if __name__ == "__main__":
    main()
