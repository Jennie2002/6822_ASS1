# Model Artifacts Input Layer

In a production deployment, this folder would contain bank-provided credit model artifacts, vendor model packages, or model prediction files submitted for governance review.

For this prototype, the model artifacts are represented as Python pickle files:

- `logistic_regression.pkl` represents a traditional scorecard baseline.
- `xgboost.pkl` represents a portable tree-boosting challenger model. It uses scikit-learn gradient boosting in this prototype so the artifact can run without requiring the external `xgboost` package.
- `fnn.pkl` represents a high-complexity AI model.
- `model_manifest.csv` records the model role, complexity, deployment gate, and artifact path.

Build or rebuild the prototype artifacts with:

```bash
python input/model_artifacts/build_model_artifacts.py
```

These pickle files are suitable for a university prototype. A production system should use controlled model registry infrastructure, signed artifacts, or model serving APIs rather than loading arbitrary pickle files.
