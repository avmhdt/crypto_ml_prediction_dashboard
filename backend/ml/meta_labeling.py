"""Meta-labeling: secondary LightGBM model.

AFML Ch.3: The secondary model predicts whether the primary model's
prediction will be correct (hit) or incorrect (miss).

Optimized for Precision — high confidence that a signal is worth acting on.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path


class MetaLabelingModel:
    """Secondary LightGBM predicting hit/miss of primary model.

    meta_label = 1 if primary_prediction == true_label else 0
    Output probability is used for bet sizing.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    @staticmethod
    def construct_meta_labels(primary_predictions: np.ndarray,
                               true_labels: np.ndarray) -> np.ndarray:
        """Construct meta-labels: 1 if primary was correct, 0 otherwise.

        Parameters
        ----------
        primary_predictions : array of {-1, 1}
        true_labels : array of {-1, 1}

        Returns
        -------
        meta_labels : array of {0, 1}
        """
        return (primary_predictions == true_labels).astype(int)

    def fit(self, X: pd.DataFrame, primary_predictions: np.ndarray,
            true_labels: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "MetaLabelingModel":
        """Train meta-labeling model.

        Parameters
        ----------
        X : feature matrix (same features as primary model)
        primary_predictions : primary model's side predictions {-1, 1}
        true_labels : actual labels {-1, 1}
        sample_weight : optional sample weights
        """
        self.feature_names = list(X.columns)
        meta_labels = self.construct_meta_labels(primary_predictions, true_labels)

        # Add primary prediction as a feature for the meta model
        X_meta = X.copy()
        X_meta["primary_side"] = primary_predictions

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_meta, meta_labels,
            sample_weight=sample_weight,
        )
        return self

    def predict_proba(self, X: pd.DataFrame,
                      primary_predictions: np.ndarray) -> np.ndarray:
        """Predict probability that primary model is correct.

        Returns P(meta=1) in [0, 1].
        """
        X_meta = X[self.feature_names].copy()
        X_meta["primary_side"] = primary_predictions
        return self.model.predict_proba(X_meta)[:, 1]

    def predict(self, X: pd.DataFrame,
                primary_predictions: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 = primary is correct, 0 = incorrect."""
        proba = self.predict_proba(X, primary_predictions)
        return (proba >= threshold).astype(int)

    def save(self, path: Path, features_path: Path | None = None) -> None:
        """Save meta-labeling model and feature names."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        if features_path is None:
            features_path = path.with_name(
                path.stem.replace("_secondary", "_meta_features") + ".json"
            )
        import json
        features_path.write_text(json.dumps(self.feature_names))

    def load(self, path: Path, features_path: Path | None = None) -> "MetaLabelingModel":
        """Load meta-labeling model and feature names."""
        self.model = joblib.load(path)
        if features_path is None:
            features_path = path.with_name(
                path.stem.replace("_secondary", "_meta_features") + ".json"
            )
        if features_path.exists():
            import json
            self.feature_names = json.loads(features_path.read_text())
        return self
