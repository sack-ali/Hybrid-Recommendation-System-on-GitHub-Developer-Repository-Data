"""Feature engineering: social, popularity, text, and edge features."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
DEV_NUMERIC_COLS = [
    "Followers",
    "Following",
    "Public Repositories",
    "starredRepoCount",
    "yearly_contributions",
]

REPO_NUMERIC_COLS = [
    "stargazers_count",
    "forks_count",
    "watching",
    "contributors_count",
    "commits_count",
    "open_issues_count",
    "size",
]

EDGE_BINARY_COLS = ["isForked", "isTopContributor"]

TEXT_COLS_FILL = ["description", "readme", "topics", "languages", "Bio"]


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def _coerce_numeric(data: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    return data


def _coerce_text(data: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str)
    return data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of the merged interaction frame.

    Adds columns:
        social_score, repo_popularity_score,
        edge_social_signal, social_score_final,
        repo_text, dev_text
    """
    data = data.copy()

    # Clean numeric/text columns
    data = _coerce_numeric(data, DEV_NUMERIC_COLS + REPO_NUMERIC_COLS)
    data = _coerce_text(data, TEXT_COLS_FILL)

    # Developer social score
    data["social_score"] = data[DEV_NUMERIC_COLS].sum(axis=1)

    # Repository popularity score
    popularity_cols = [
        "stargazers_count",
        "forks_count",
        "watching",
        "contributors_count",
        "commits_count",
    ]
    data["repo_popularity_score"] = data[popularity_cols].sum(axis=1)

    # Edge features
    for col in EDGE_BINARY_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
    data["edge_social_signal"] = data["isForked"] + data["isTopContributor"]
    data["social_score_final"] = data["social_score"] + data["edge_social_signal"]

    # Text fields for modeling
    data["repo_text"] = (
        data["description"].fillna("") + " "
        + data["topics"].fillna("") + " "
        + data["languages"].fillna("")
    )
    data["dev_text"] = data["Bio"].fillna("")

    return data


def normalize_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize the three scoring components into [0, 1].

    Expects `technical_score`, `social_score_final`, and
    `repo_popularity_score` to already be present.
    """
    data = data.copy()
    scaler = MinMaxScaler()

    data["technical_norm"] = scaler.fit_transform(data[["technical_score"]])
    data["social_norm"] = scaler.fit_transform(data[["social_score_final"]])
    data["repo_popularity_norm"] = scaler.fit_transform(data[["repo_popularity_score"]])

    return data


def interaction_matrix_stats(data: pd.DataFrame) -> dict:
    """Compute user-item interaction matrix diagnostics.

    Uses a binary interaction indicator (1 if (dev, repo) exists in `data`).
    """
    matrix = data.pivot_table(
        index="dev_id",
        columns="repo_id",
        values="isForked",
        aggfunc="max",
        fill_value=0,
    ).clip(upper=1)

    total_cells = matrix.shape[0] * matrix.shape[1]
    non_zero = int((matrix.values > 0).sum())
    density = non_zero / total_cells if total_cells else 0.0

    return {
        "shape": matrix.shape,
        "non_zero": non_zero,
        "density": density,
        "sparsity": 1 - density,
        "interactions_per_dev": matrix.sum(axis=1),
        "interactions_per_repo": matrix.sum(axis=0),
    }
