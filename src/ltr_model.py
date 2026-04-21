"""Random Forest Learning-to-Rank model."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from . import config
from .sampling import FEATURE_COLS


def train_ltr_model(
    ranking_data: pd.DataFrame,
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int = config.RF_MAX_DEPTH,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """Train a Random Forest on the ranking dataset.

    Returns
    -------
    model : RandomForestClassifier
    ranking_data : DataFrame with an added `ltr_score` column.
    """
    X = ranking_data[FEATURE_COLS]
    y = ranking_data["label"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=-1,
    )
    model.fit(X, y)

    ranking_data = ranking_data.copy()
    ranking_data["ltr_score"] = model.predict_proba(X)[:, 1]

    return model, ranking_data


def evaluate_ltr_model(model: RandomForestClassifier, ranking_data: pd.DataFrame) -> dict:
    """Return classification report and ROC-AUC on the ranking dataset."""
    X = ranking_data[FEATURE_COLS]
    y = ranking_data["label"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "classification_report_text": classification_report(y, y_pred),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


def feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    """Return the MDI feature importances as a sorted DataFrame."""
    return (
        pd.DataFrame(
            {"Feature": FEATURE_COLS, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
