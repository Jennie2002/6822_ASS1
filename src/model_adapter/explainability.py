import pandas as pd


def compute_lime_explanation(
    model: object,
    X_train: pd.DataFrame,
    applicant: pd.Series,
    feature_columns: list[str],
    num_features: int = 6,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None, "LIME is not installed. Install it with `pip install lime`."

    explainer = LimeTabularExplainer(
        training_data=X_train[feature_columns].values,
        feature_names=feature_columns,
        class_names=["Bad / default risk", "Good / repaid"],
        mode="classification",
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(
        applicant[feature_columns].values,
        model.predict_proba,
        num_features=num_features,
    )
    return (
        pd.DataFrame(explanation.as_list(), columns=["feature_rule", "contribution"]),
        None,
    )


def compute_shap_local_explanation(
    model: object,
    X_train: pd.DataFrame,
    applicant: pd.Series,
    feature_columns: list[str],
    max_background_rows: int = 20,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed. Install it with `pip install shap`."

    background = X_train[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    background = background.sample(
        min(max_background_rows, len(X_train)), random_state=42
    )
    applicant_frame = (
        applicant[feature_columns]
        .to_frame()
        .T
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    try:
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(applicant_frame)
        values = shap_values.values

        if values.ndim == 3:
            contributions = values[0, :, 1]
        else:
            contributions = values[0]

        result = pd.DataFrame(
            {
                "feature": feature_columns,
                "shap_contribution": contributions,
                "feature_value": applicant_frame.iloc[0].values,
            }
        )
        result["absolute_contribution"] = result["shap_contribution"].abs()
        result = result.sort_values("absolute_contribution", ascending=False)
        return result, None
    except Exception as exc:
        return None, f"SHAP explanation could not be computed for this model: {exc}"
