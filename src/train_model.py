"""
Baseline vs trained model on a real public dataset.

Dataset: Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn
(569 samples, 30 numeric features, binary target). No download required.

This is a benchmark for showing the method: split, scale, train, compare
against a majority-class baseline, and report honest metrics. It is NOT a
diagnostic tool and must not be used for medical decisions.

Usage:
    python src/train_model.py
"""

from __future__ import annotations

import sys
import warnings

# scikit-learn 1.6.x + scipy 1.18.x emit an OptimizeWarning about an unknown
# solver option "iprint". It is harmless here; suppress it so the output stays
# readable and the exit code is clean.
warnings.filterwarnings(
    "ignore", message=".*Unknown solver options: iprint.*", category=Warning
)

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bundled dataset into a DataFrame plus a target Series."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 1 = malignant, 0 = benign
    return X, y


def build_model() -> Pipeline:
    """
    Scaling and model are wrapped in one Pipeline so the scaler is fitted on
    training data only. This is what prevents data leakage.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    """Compute the standard classification metrics for one model."""
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def main() -> int:
    X, y = load_dataset()
    print(f"dataset : {X.shape[0]} rows, {X.shape[1]} features")
    print(f"classes : {dict(y.value_counts().sort_index())}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"split   : train={len(X_train)}, test={len(X_test)}")
    print()

    # Majority-class baseline: shows how much the model actually adds.
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    base_metrics = evaluate("baseline (majority class)", y_test, baseline.predict(X_test))

    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate("logistic regression", y_test, model.predict(X_test), proba)

    # 5-fold cross-validation on the training set only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_model(), X_train, y_train, cv=cv, scoring="f1")

    rows = [base_metrics, model_metrics]
    table = pd.DataFrame(rows).set_index("model")
    print(table.round(4).to_string())
    print()
    print(
        f"5-fold CV F1 on train: mean={cv_scores.mean():.4f} "
        f"std={cv_scores.std():.4f} folds={[round(s, 4) for s in cv_scores]}"
    )
    print()
    print("confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print()
    print(
        f"accuracy gain over baseline: "
        f"{model_metrics['accuracy'] - base_metrics['accuracy']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
