"""Linear weighted combination of content, social and popularity scores."""

from __future__ import annotations

import pandas as pd

from . import config


def add_linear_score(
    data: pd.DataFrame,
    w_content: float = config.W_CONTENT,
    w_social: float = config.W_SOCIAL,
    w_popularity: float = config.W_POPULARITY,
) -> pd.DataFrame:
    """Compute the hybrid linear score.

    Formula:
        linear_score = w_content * technical_norm
                     + w_social  * social_norm
                     + w_popularity * repo_popularity_norm
    """
    assert abs((w_content + w_social + w_popularity) - 1.0) < 1e-6, (
        "Linear combination weights must sum to 1."
    )

    data = data.copy()
    data["linear_score"] = (
        w_content * data["technical_norm"]
        + w_social * data["social_norm"]
        + w_popularity * data["repo_popularity_norm"]
    )
    return data
