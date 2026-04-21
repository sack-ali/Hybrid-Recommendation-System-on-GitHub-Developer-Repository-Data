"""TF-IDF vectorization and cosine similarity between developers and repos."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def build_repo_tfidf(data: pd.DataFrame) -> tuple[TfidfVectorizer, "np.ndarray"]:
    """Fit a TF-IDF vectorizer on repo_text and return the sparse matrix."""
    vectorizer = TfidfVectorizer(
        max_features=config.REPO_TFIDF_MAX_FEATURES,
        stop_words=config.TFIDF_STOP_WORDS,
    )
    tfidf_matrix = vectorizer.fit_transform(data["repo_text"])
    return vectorizer, tfidf_matrix


def build_dev_tfidf(data: pd.DataFrame) -> tuple[TfidfVectorizer, "np.ndarray"]:
    """Fit a TF-IDF vectorizer on dev_text (developer Bio) and return the matrix."""
    vectorizer = TfidfVectorizer(
        max_features=config.DEV_TFIDF_MAX_FEATURES,
        stop_words=config.TFIDF_STOP_WORDS,
    )
    tfidf_matrix = vectorizer.fit_transform(data["dev_text"])
    return vectorizer, tfidf_matrix


def build_developer_profiles(data: pd.DataFrame, repo_tfidf) -> Dict[object, np.ndarray]:
    """Aggregate a developer's interacted repositories into a single TF-IDF centroid."""
    profiles: Dict[object, np.ndarray] = {}
    dev_ids = data["dev_id"].values

    for dev in pd.unique(dev_ids):
        idx = np.where(dev_ids == dev)[0]
        if len(idx) > 0:
            profiles[dev] = np.asarray(repo_tfidf[idx].mean(axis=0))
    return profiles


def compute_technical_scores(
    data: pd.DataFrame,
    repo_tfidf,
    dev_profiles: Dict[object, np.ndarray],
) -> pd.Series:
    """Cosine similarity between each developer profile and the current repo vector.

    Returns a Series aligned with `data.index`.
    """
    scores = np.empty(len(data), dtype=float)

    for i, dev in enumerate(data["dev_id"].values):
        dev_vec = dev_profiles[dev]
        repo_vec = repo_tfidf[i].toarray()
        scores[i] = cosine_similarity(dev_vec, repo_vec)[0][0]

    return pd.Series(scores, index=data.index, name="technical_score")
