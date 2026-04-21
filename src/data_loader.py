"""Load the three source CSVs and merge them into a single interaction frame."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def load_raw(
    developers_path: Path = config.DEVELOPERS_CSV,
    repositories_path: Path = config.REPOSITORIES_CSV,
    edges_path: Path = config.EDGES_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV files.

    Returns
    -------
    developers : pd.DataFrame
    repositories : pd.DataFrame
    edges : pd.DataFrame
    """
    developers = pd.read_csv(developers_path)
    repositories = pd.read_csv(repositories_path)
    edges = pd.read_csv(edges_path)
    return developers, repositories, edges


def merge_interactions(
    developers: pd.DataFrame,
    repositories: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Merge edges with developer and repository metadata.

    The result has one row per observed (dev_id, repo_id) interaction,
    enriched with every developer and repository column.
    """
    data = edges.merge(developers, on="dev_id", how="left")
    data = data.merge(repositories, on="repo_id", how="left")
    return data


def load_and_merge() -> pd.DataFrame:
    """Convenience helper: load + merge in one call."""
    developers, repositories, edges = load_raw()
    return merge_interactions(developers, repositories, edges)
