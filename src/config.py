"""Project-wide configuration: paths, hyperparameters, random seeds."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DEVELOPERS_CSV = DATA_DIR / "developers.csv"
REPOSITORIES_CSV = DATA_DIR / "repositories.csv"
EDGES_CSV = DATA_DIR / "edgelist.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Linear weighted combination
# ---------------------------------------------------------------------------
# Weights must sum to 1.0
W_CONTENT = 0.5     # technical / content similarity weight
W_SOCIAL = 0.3      # collaborative / social weight
W_POPULARITY = 0.2  # popularity weight

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
REPO_TFIDF_MAX_FEATURES = 500
DEV_TFIDF_MAX_FEATURES = 300
TFIDF_STOP_WORDS = "english"

# ---------------------------------------------------------------------------
# Random Forest Learning-to-Rank
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_CLASS_WEIGHT = "balanced"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# Ratio of negative to positive samples (1.0 = balanced)
NEGATIVE_RATIO = 1.0
