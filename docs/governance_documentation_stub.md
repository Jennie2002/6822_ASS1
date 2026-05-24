# Governance Documentation Stub

## 1. Tool Purpose

This prototype is an HSBC-style US/EU credit model governance decision-support tool. It is designed to help internal model risk, compliance, legal, and business governance teams assess whether a credit scoring model is governable under a selected jurisdiction profile.

The tool is not a generic machine learning leaderboard. It does not automatically approve, reject, or deploy models. Its purpose is to convert model evidence into governance findings:

- MRM validation judgement
- multi-layer fairness governance results
- explainability evidence status
- jurisdiction-specific governance flags
- required governance actions
- escalation owners
- governance posture
- human committee review note

Final deployment judgement remains with human governance committees, compliance, legal, and senior management.

## 2. What The Tool Does

The Streamlit dashboard:

- reads one fixed benchmark governance dataset from `input/datasets/processed/combined_credit_dataset.csv`
- evaluates three prototype model artifacts: Logistic Regression, XGBoost, and FNN
- calculates MRM performance evidence: accuracy, precision, recall, ROC-AUC, and KS
- converts model probabilities into credit score, risk band, and manual review triggers
- performs explicit fairness-group audits for `age_group`, `income_group`, and `synthetic_protected_group`
- applies US/EU jurisdiction-specific governance profiles
- applies BISG-weighted proxy monitoring for the US as supplementary aggregate fair-lending monitoring
- disables BISG-style proxy demographic inference for the EU profile
- checks SHAP/LIME explainability evidence status where available
- generates governance flags, required actions, escalation owners, and governance posture
- exports a governance assessment report

The governance posture values are limited to:

- `Deployable with controls`
- `Enhanced review required`
- `Deployment hold recommended`

The tool intentionally avoids language such as approved, fully compliant, automatically accepted, or AI-approved.

## 3. Data Inputs and Assumptions

The processed dataset is built from two benchmark credit datasets:

- Give Me Some Credit
- German Credit Risk

The combined dataset contains:

- 20,000 Give Me Some Credit rows
- 1,000 German Credit rows
- 5,000 conditional fusion rows

The purpose of the fused dataset is not to claim that all records are observed real customers. The purpose is to create a richer governance testbed that preserves useful patterns from both benchmark datasets.

Key assumptions:

- Benchmark credit data can approximate a university-level credit governance scenario.
- The dataset is not real HSBC customer data.
- `synthetic_protected_group` is a simulated protected-class label for aggregate fairness testing.
- Faker-generated surname, postcode, geography, census tract, and BISG probability fields are governance support fields only.
- BISG-style fields are used only for aggregate US proxy monitoring, not for individual credit decisions.
- SHAP evidence is treated as prototype-level explainability evidence, not as a complete legal explanation.
- Simplified thresholds represent governance triggers, not formal legal determinations.

## 4. Model Layer

The model layer evaluates three model artifacts:

| Model | Governance role | Complexity | Governance meaning |
|---|---|---|---|
| Logistic Regression | Traditional scorecard baseline | Low | Easier to explain, but still subject to fairness and validation review |
| XGBoost | Challenger machine learning model | Medium | May offer stronger performance but requires SHAP/LIME-style explainability evidence |
| FNN | High-complexity AI model | High | Higher opacity and stronger governance burden |

The tool treats model performance as MRM evidence only. Accuracy, ROC-AUC, KS, precision, and recall do not determine deployment by themselves.

## 5. Model Risk Management Logic

The model risk layer converts performance evidence into an MRM validation judgement:

| Condition | MRM validation |
|---|---|
| ROC-AUC >= 0.75 and KS >= 0.30 | Pass |
| ROC-AUC >= 0.65 and KS >= 0.20 | Review required |
| Otherwise | Fail |

The model risk layer also provides:

- drift detection prototype signal
- human review escalation signal
- fairness/bias monitoring judgement
- explainability risk signal

This layer supports model governance review. It does not replace independent model validation.

## 6. Fairness Governance Logic

The tool performs explicit fairness-group audits for:

- `age_group`
- `income_group`
- `synthetic_protected_group`

For each group, the tool calculates:

- approval rate by group
- approval disparity
- false positive rate disparity
- false negative rate disparity
- governance status: `Pass` or `Review required`

If any of the disparity metrics exceeds the selected disparity threshold, the fairness group is marked as `Review required`.

The governance engine uses the worst-case disparity across the explicit fairness groups.

## 7. Jurisdiction Logic

### EU Governance Profile

The EU profile represents a high-risk AI credit scoring governance setting.

EU logic:

- applies explicit fairness-group monitoring
- treats creditworthiness assessment as high-risk AI
- requires bias testing, explainability evidence, audit trail, and human oversight
- does not use BISG-style inferred race or ethnicity proxy monitoring

Governance interpretation:

```text
The EU governance profile applies explicit fairness-group monitoring without BISG-based proxy demographic inference.
```

### US Governance Profile

The US profile represents fair-lending and model risk governance.

US logic:

- applies explicit fairness-group monitoring
- uses BISG-weighted proxy monitoring as supplementary aggregate fair-lending monitoring
- frames disparity as fair-lending, litigation, and reputational risk
- requires adverse-action explanation support for declined or borderline cases

Governance interpretation:

```text
The US governance profile supplements explicit fairness-group monitoring with BISG-weighted proxy disparate-impact monitoring.
```

Important boundary:

```text
BISG-weighted monitoring is used only for aggregate governance review and fairness-risk assessment. It is not used for individual credit decisions.
```

## 8. Explainability Evidence

Explainability evidence is used as governance evidence, not as an automatic approval mechanism.

| Model | Evidence logic |
|---|---|
| Logistic Regression | Native coefficient interpretability available |
| XGBoost | SHAP evidence is generated where possible; LIME is optional / not generated in the current prototype |
| FNN | No validated SHAP/LIME evidence artifact is generated for FNN in the current prototype; this intentionally triggers enhanced explainability governance review. |

If explanation evidence is missing for a complex model, the tool can trigger:

- `EXPLAINABILITY_REVIEW_REQUIRED`
- `EXPLAINABILITY_EVIDENCE_MISSING`
- `generate_shap_lime_evidence`
- Model Validation Team escalation

## 9. Governance Flags and Triggers

The governance action layer converts risk signals into structured governance flags.

| Flag | Trigger | Severity | Owner |
|---|---|---|---|
| `EU_HIGH_RISK_AI_APPLIES` | jurisdiction is EU | High | AI Governance Committee |
| `HUMAN_OVERSIGHT_REQUIRED` | EU profile or high-risk automated rejection | Medium | Business Owner + Compliance |
| `FAIRNESS_THRESHOLD_BREACH` | approval/FPR/FNR disparity exceeds threshold | High | Fair Lending / Compliance Committee |
| `FAIRNESS_SEVERE_BREACH` | approval disparity exceeds severe threshold | High | Fair Lending / Compliance Committee |
| `EXPLAINABILITY_REVIEW_REQUIRED` | EU and model complexity is Medium or High | Medium | Model Risk Committee |
| `EXPLAINABILITY_EVIDENCE_MISSING` | explanation evidence status is Missing | High for EU, Medium for US | Model Validation Team |
| `MODEL_RISK_COMMITTEE_REVIEW_REQUIRED` | conditional/restricted deployment gate or MRM fail | Medium | Model Risk Committee |
| `DEPLOYMENT_HOLD_RECOMMENDED` | final posture is deployment hold | High | Senior Governance Committee |

## 10. Required Governance Actions

Flags map to governance actions:

| Trigger | Required action |
|---|---|
| EU high-risk AI applies | `complete_bias_testing_evidence` |
| Fairness threshold breach | `conduct_fairness_investigation` |
| Explainability review required | `complete_explainability_review` |
| Explanation evidence missing | `generate_shap_lime_evidence` |
| Human oversight required | `document_human_oversight_process` |
| Model risk committee review required | `obtain_model_risk_committee_signoff` |
| Deployment hold recommended | `complete_governance_remediation_before_reassessment` |
| Default monitoring | `monitor_post_deployment_drift` |

## 11. Governance Posture Logic

The tool outputs a governance posture, not a deployment approval.

`Deployment hold recommended` can be triggered by:

- severe fairness breach
- MRM validation fail
- EU + high-complexity model + missing explanation evidence
- EU + fairness breach + medium/high complexity + missing explanation evidence

`Enhanced review required` can be triggered by:

- fairness governance review required
- high explainability burden
- EU medium/high complexity with missing or limited explanation evidence

`Deployable with controls` can be shown when:

- MRM validation passes
- no higher-severity governance trigger requires enhanced review or hold

This posture is a governance assessment output only. It does not approve deployment.

## 12. What The Tool Does Not Do

The tool does not:

- replace legal advice
- determine the legally correct jurisdiction automatically
- prove absence of discrimination
- approve or reject model deployment automatically
- make individual credit decisions
- use BISG for individual credit decisions
- generate production-ready adverse action notices
- retrain or remediate models
- provide a full audit log
- replace independent model validation
- replace human governance committees

## 13. Where The Tool Can Fail

Important failure modes:

- Benchmark data may not represent HSBC customer portfolios.
- Conditional fusion rows are useful for demonstration but are not observed real customers.
- `synthetic_protected_group` may not reflect real protected-class distributions.
- BISG-style proxy monitoring can be inaccurate or ethically sensitive.
- Faker-generated surname and geography fields do not represent real census reference data.
- SHAP/LIME explanations can be unstable or incomplete.
- Feature importance is not a complete legal explanation for an individual credit denial.
- Simplified US/EU rules do not capture the full regulatory environment.
- Fixed disparity thresholds may not match real institutional risk appetite.
- Model artifacts are prototype models, not production-validated bank models.
- The dashboard can identify governance issues but cannot remediate them.

## 14. Intentional Design Choices

### Active Addition

The tool includes a Governance Action Layer. This layer is not only a reporting feature. It converts risk signals into operational governance actions, such as:

- manual review escalation
- enhanced explainability review
- fair-lending investigation
- governance committee sign-off
- deployment hold recommendation

This was added because static compliance reporting alone does not reduce real operational AI risk.

### Intentional Non-Feature

The tool intentionally avoids making large static explainability documentation packages the primary output. Instead, it focuses on governance-oriented signals, triggers, and escalation prompts.

This design choice avoids prioritising formal documentation over substantive governance intervention.

## 15. Senior Management Summary

This prototype is best understood as a policy-driven AI governance assessment portal for a multinational bank. It demonstrates how the same model can create different governance obligations under US and EU profiles.

The US configuration extends fairness governance through supplementary BISG-weighted proxy monitoring, while the EU profile restricts governance monitoring to explicit fairness-group analysis without proxy demographic inference.

The tool is useful for identifying:

- which models need enhanced review
- which governance flags were triggered
- which committees or owners should be involved
- which controls must be completed before deployment can be considered

It is not useful for proving that a model is legally compliant or ready for production deployment.

## 16. Future Improvements

With more time and better data access, the tool could add:

- real institution-grade credit portfolio data
- real protected-class or legally approved proxy reference data
- production audit logging
- adverse action reason generation
- formal model validation workflow
- human review case management
- model remediation recommendations
- threshold sensitivity analysis
- jurisdiction-specific legal review workflow
- documented owner sign-off records
