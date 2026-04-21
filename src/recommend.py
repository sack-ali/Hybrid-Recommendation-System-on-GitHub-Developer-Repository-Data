"""Top-K recommendation helpers using both linear and LTR scores."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def recommend_for_developer(
    dev_id,
    data: pd.DataFrame,
    ranking_data: pd.DataFrame,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-N recommendations for a given developer.

    Parameters
    ----------
    dev_id : value in data["dev_id"]
    data : DataFrame with `linear_score` per (dev_id, repo_id).
    ranking_data : DataFrame with `ltr_score` per (dev_id, repo_id).
    top_n : number of recommendations to return per model.

    Returns
    -------
    linear_rec : top-N by linear score.
    ltr_rec : top-N by learning-to-rank score.
    """
    linear_rec = (
        data.loc[data["dev_id"] == dev_id, ["repo_id", "linear_score"]]
        .drop_duplicates("repo_id")
        .sort_values("linear_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ltr_rec = (
        ranking_data.loc[ranking_data["dev_id"] == dev_id, ["repo_id", "ltr_score", "label"]]
        .drop_duplicates("repo_id")
        .sort_values("ltr_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return linear_rec, ltr_rec


def attach_repo_names(
    recommendations: pd.DataFrame,
    repositories: pd.DataFrame,
) -> pd.DataFrame:
    """Join human-readable repo metadata onto a recommendation frame."""
    cols = [
        c
        for c in [
            "repo_id",
            "repo_name",
            "owner_username",
            "stargazers_count",
            "languages",
            "topics",
        ]
        if c in repositories.columns
    ]
    return recommendations.merge(repositories[cols], on="repo_id", how="left")
