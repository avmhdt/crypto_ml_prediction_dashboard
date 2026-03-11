"""Primary LightGBM model for side prediction.

Optimized for Recall — catch as many true signals as possible.
False positives are filtered downstream by meta-labeling.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class PrimaryModel:
    """LightGBM classifier predicting side {-1, 1}."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model: lgb.LGBMClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "PrimaryModel":
        """Train the primary model.

        Parameters
        ----------
        X : DataFrame of features
        y : array of labels in {-1, 1}
        sample_weight : AFML sample weights (uniqueness * attribution * decay)
        """
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Convert labels: {-1, 1} → {0, 1} for LightGBM binary classification
        y_binary = ((y + 1) / 2).astype(int)

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_scaled, y_binary,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict side: returns array of {-1, 1}."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        y_binary = self.model.predict(X_scaled)
        # Convert back: {0, 1} → {-1, 1}
        return (y_binary * 2 - 1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class (side=1)."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        )
        return imp.sort_values(ascending=False)

    def save(self, primary_path: Path, scaler_path: Path,
             features_path: Path) -> None:
        """Save model, scaler, and feature list."""
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, primary_path)
        joblib.dump(self.scaler, scaler_path)
        import json
        features_path.write_text(json.dumps(self.feature_names))

    def load(self, primary_path: Path, scaler_path: Path,
             features_path: Path) -> "PrimaryModel":
        """Load model, scaler, and feature list."""
        self.model = joblib.load(primary_path)
        self.scaler = joblib.load(scaler_path)
        import json
        self.feature_names = json.loads(features_path.read_text())
        return self
