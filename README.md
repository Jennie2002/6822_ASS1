# HSBC US/EU Credit Model Governance Dashboard

Name: ZHENJIGNWEN

Matriculation ID: G2508036E

Email: ZHEN0168@e.ntu.edu.sg

## Overview

This Streamlit prototype supports Task 3 of the assignment. It demonstrates an internal HSBC-style US/EU AI credit model governance decision-support portal, based on the HSBC / AI governance / credit scoring topic developed in Task 1 and Task 2.

The tool:

- uses a fixed benchmark governance dataset for demonstration
- evaluates three prototype credit model artifacts in sequence
- computes accuracy, precision, recall, ROC-AUC, and KS
- computes fairness metrics by group
- includes a model risk management layer for validation, explainability, fairness, drift monitoring, and human review escalation
- applies simplified jurisdiction rules for the US and EU
- converts risk findings into structured governance flags, required actions, escalation owners, and governance posture
- treats feature importance, SHAP, and LIME as explainability evidence for governance review
- generates a jurisdiction-specific governance dashboard and action summary
- includes a concise governance stub required by the assignment

The design goal is substantive governance rather than formalistic compliance. Static compliance reporting alone does not reduce real operational AI risk, so the tool focuses on governance triggers, human review prompts, escalation actions, and deployment restriction flags. The system does not automatically approve or reject models; final deployment judgement remains with HSBC governance committees.

Fairness group options include `age_group`, `income_group`, and `synthetic_protected_group`. The benchmark credit datasets do not contain complete real protected-class attributes, so `synthetic_protected_group` is a simulated protected-class label for aggregate fairness testing. In the US configuration, BISG-style soft weighting can additionally be applied as supplementary proxy monitoring. These variables are used only for aggregate governance monitoring, not for individual credit decisions.

The EU configuration applies explicit fairness-group monitoring without BISG-based proxy demographic inference. The US configuration extends explicit fairness governance through supplementary BISG-weighted proxy disparate-impact monitoring.

## Project Structure

```text
ASS1/
  app.py                        # Static Streamlit governance report generator
  requirements.txt              # Python dependencies
  README.md                     # Project notes and run instructions
  input/
    datasets/
      raw/
        give_me_some_credit_data.csv # Benchmark source: Give Me Some Credit
        german_credit_data.csv       # Benchmark source: German Credit Risk
      processed/
        combined_credit_dataset.csv  # Final fixed governance-ready dataset
      build/
        data_fusion.py               # Offline conditional data fusion logic
        build_combined_dataset.py    # Rebuilds the processed dataset
    model_artifacts/
      logistic_regression.pkl        # Prototype scorecard baseline artifact
      xgboost.pkl                    # Prototype challenger model artifact
      fnn.pkl                        # Prototype high-complexity model artifact
      model_manifest.csv             # Artifact registry and model governance metadata
      build_model_artifacts.py       # Rebuilds prototype model artifacts
      README.md                      # Model artifact input notes
  docs/
    governance_documentation_stub.md # Consolidated governance documentation
  src/
    config/
      settings.py               # Shared constants
      model_profiles.yaml       # Model role, complexity, explainability, and deployment gate
      us_config.yaml            # US jurisdiction configuration
      eu_config.yaml            # EU jurisdiction configuration
      thresholds.yaml           # Prototype governance thresholds
    io/
      dataset_loader.py         # Runtime reader for processed dataset
    model_adapter/
      credit_model.py           # Model artifact loading, prediction, explanation
      scorecard.py              # PD, score bands, decision recommendation
      explainability.py         # SHAP and LIME local explanation helpers
    model_risk/
      layer.py                  # Model validation, fairness, drift, human review
    jurisdiction/
      engine.py                 # US/EU jurisdiction rules engine and config loader
    governance/
      action_layer.py           # Governance action and deployment flag layer
    reporting/
      governance_report.py      # Governance report and action summary
```

This structure keeps the prototype simple while separating the main assignment concerns: data, model, governance rules, and user interface.

## Included Data

The app uses one fixed governance-ready dataset:

```text
input/datasets/processed/combined_credit_dataset.csv
```

It is built from two benchmark credit datasets plus conditional fusion rows and generated governance support fields such as BISG-style proxy columns:

- `input/datasets/raw/give_me_some_credit_data.csv` for Give Me Some Credit
- `input/datasets/raw/german_credit_data.csv` for German Credit Risk

Current combined dataset composition:

```text
give_me_some_credit    20000 rows
conditional_fusion      5000 rows
german_credit           1000 rows
```

To rebuild the final combined CSV:

```bash
python input/datasets/build/build_combined_dataset.py
```

To rebuild the prototype model artifacts:

```bash
python input/model_artifacts/build_model_artifacts.py
```

## Supporting Documentation

The `docs/` folder contains one consolidated governance documentation stub:

- `governance_documentation_stub.md`: explains what the tool does, what it assumes, where it can fail, what it does not do, and what could be improved with more time or better data.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

The dependency versions are pinned for reproducibility with the included fixed dataset and model artifacts.

Run the governance dashboard for the EU:

```bash
streamlit run app.py -- --jurisdiction EU
```

Run the governance dashboard for the US:

```bash
streamlit run app.py -- --jurisdiction US
```

US runs enable BISG soft probability weighting by default. To disable it:

```bash
streamlit run app.py -- --jurisdiction US --no-bisg
```

Alternative command if Streamlit is not on PATH:

```bash
python -m streamlit run app.py -- --jurisdiction EU
```

## Important Limitations

This is an assignment prototype, not legal advice and not a production credit decisioning system.

Important limitations:

- the data is benchmark-style harmonised data with generated governance support fields
- no real HSBC customer data is used
- the legal rules are simplified
- feature importance, SHAP, and LIME are governance-support mechanisms only and do not constitute complete legal explanations
- the prototype does not generate customer adverse action notices
- the prototype does not include a full audit log
- the prototype does not include production model validation

## Future Improvements

With more time, the project could add:

- audit logs
- customer-facing adverse action reasons
- human review workflow
- jurisdiction-specific threshold profiles
- stronger model validation and stress testing
- additional jurisdiction-specific governance profiles
