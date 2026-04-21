"""Positive / negative sampling for learning-to-rank."""

from __future__ import annotations

import random

import pandas as pd

from . import config


FEATURE_COLS = ["technical_norm", "social_norm", "repo_popularity_norm"]


def build_ranking_dataset(
    data: pd.DataFrame,
    negative_ratio: float = config.NEGATIVE_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Construct a balanced dataset of positive and negative (dev, repo) samples.

    Positive samples are the observed (dev_id, repo_id) pairs in `data`.
    Negative samples are random (dev_id, repo_id) pairs that do NOT appear
    in `data`; their features use the developer's mean scores and the repo's
    mean popularity, which gives the classifier a realistic but unobserved
    profile to learn against.
    """
    random.seed(random_seed)

    # Positives
    positive = data[["dev_id", "repo_id"] + FEATURE_COLS].copy()
    positive["label"] = 1

    # Negatives
    all_devs = data["dev_id"].unique()
    all_repos = data["repo_id"].unique()
    existing_pairs = set(zip(data["dev_id"], data["repo_id"]))

    target_negatives = int(len(positive) * negative_ratio)
    rows = []

    # Precompute developer and repo averages once for efficiency
    dev_means = (
        data.groupby("dev_id")[["technical_norm", "social_norm"]].mean()
    )
    repo_pop = data.groupby("repo_id")["repo_popularity_norm"].mean()

    attempts = 0
    max_attempts = target_negatives * 50  # safety valve
    while len(rows) < target_negatives and attempts < max_attempts:
        attempts += 1
        dev = random.choice(all_devs)
        repo = random.choice(all_repos)
        if (dev, repo) in existing_pairs:
            continue

        tech_val = dev_means.loc[dev, "technical_norm"] if dev in dev_means.index else 0.0
        social_val = dev_means.loc[dev, "social_norm"] if dev in dev_means.index else 0.0
        pop_val = repo_pop.loc[repo] if repo in repo_pop.index else 0.0

        rows.append([dev, repo, tech_val, social_val, pop_val, 0])

    negative = pd.DataFrame(
        rows,
        columns=["dev_id", "repo_id"] + FEATURE_COLS + ["label"],
    )

    ranking_data = pd.concat([positive, negative], ignore_index=True)
    return ranking_data
