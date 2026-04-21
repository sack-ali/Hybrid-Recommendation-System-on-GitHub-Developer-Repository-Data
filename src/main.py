"""Command-line entry point that wires the full pipeline together.

Usage
-----
    python -m src.main --top-n 5
    python -m src.main --top-n 10 --dev-id some_dev_id
"""

from __future__ import annotations

import argparse
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config
from .data_loader import load_and_merge, load_raw
from .features import engineer_features, normalize_scores, interaction_matrix_stats
from .linear_model import add_linear_score
from .ltr_model import train_ltr_model, evaluate_ltr_model, feature_importance
from .recommend import attach_repo_names, recommend_for_developer
from .sampling import build_ranking_dataset
from .similarity import (
    build_developer_profiles,
    build_repo_tfidf,
    compute_technical_scores,
)


def build_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, RandomForestClassifier]:
    """Run the full training pipeline and return the artifacts.

    Returns
    -------
    data : DataFrame with engineered features, normalized scores, and linear_score.
    ranking_data : Positive/negative samples with ltr_score.
    ltr_model : Trained Random Forest classifier.
    """
    # 1. Load + merge
    data = load_and_merge()

    # 2. Feature engineering
    data = engineer_features(data)

    # 3. TF-IDF + developer profiles + technical scores
    _, repo_tfidf = build_repo_tfidf(data)
    dev_profiles = build_developer_profiles(data, repo_tfidf)
    data["technical_score"] = compute_technical_scores(data, repo_tfidf, dev_profiles)

    # 4. Normalize scoring components
    data = normalize_scores(data)

    # 5. Linear hybrid score
    data = add_linear_score(data)

    # 6. Positive/negative sampling
    ranking_data = build_ranking_dataset(data)

    # 7. Train Random Forest LTR
    ltr_model, ranking_data = train_ltr_model(ranking_data)

    return data, ranking_data, ltr_model


def _print_report(data: pd.DataFrame, ranking_data: pd.DataFrame, ltr_model) -> None:
    stats = interaction_matrix_stats(data)
    print("\n=== Interaction Matrix ===")
    print(f"Shape    : {stats['shape']}")
    print(f"Non-zero : {stats['non_zero']}")
    print(f"Density  : {stats['density']:.4f}")
    print(f"Sparsity : {stats['sparsity']:.4f}")

    print("\n=== Learning-to-Rank Evaluation ===")
    metrics = evaluate_ltr_model(ltr_model, ranking_data)
    print(metrics["classification_report_text"])
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")

    print("\n=== Feature Importance ===")
    print(feature_importance(ltr_model).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the hybrid recommender and print top-N recommendations."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of recommendations to return per model (default: 5).",
    )
    parser.add_argument(
        "--dev-id",
        type=str,
        default=None,
        help="Developer id to recommend for. Defaults to the first developer in the data.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the evaluation/feature-importance report.",
    )
    args = parser.parse_args()

    data, ranking_data, ltr_model = build_pipeline()

    if not args.no_report:
        _print_report(data, ranking_data, ltr_model)

    dev_id = args.dev_id if args.dev_id is not None else data["dev_id"].iloc[0]
    linear_rec, ltr_rec = recommend_for_developer(
        dev_id=dev_id,
        data=data,
        ranking_data=ranking_data,
        top_n=args.top_n,
    )

    # Attach repo names for readability
    _, repositories, _ = load_raw()
    linear_named = attach_repo_names(linear_rec, repositories)
    ltr_named = attach_repo_names(ltr_rec, repositories)

    print(f"\n=== Top-{args.top_n} recommendations for developer: {dev_id} ===")
    print("\n-- Linear hybrid score --")
    print(linear_named.to_string(index=False))
    print("\n-- Random Forest Learning-to-Rank --")
    print(ltr_named.to_string(index=False))


if __name__ == "__main__":
    main()
