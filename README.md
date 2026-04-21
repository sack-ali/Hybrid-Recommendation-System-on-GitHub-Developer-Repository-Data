# Hybrid Recommendation System on GitHub Developer–Repository Data

A hybrid recommendation system that combines **content-based filtering**, **collaborative / social signals**, and **popularity-based scoring** to recommend GitHub repositories to developers. A Random Forest **Learning-to-Rank (LTR)** model is trained on top of these signals for ranking.

This project was built as the Part A midterm deliverable for the *Machine Learning & Recommendation Systems* course and implements **11 core concepts** of modern recommender systems end-to-end on a GitHub bipartite graph dataset (Developers ↔ Repositories).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Concepts Implemented](#concepts-implemented)
- [Results](#results)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

The system predicts and ranks repositories that are most likely to be of interest to a given developer, based on:

1. **Content similarity** — TF-IDF over repo descriptions, topics, and languages, matched against an aggregated developer profile.
2. **Social / interaction signals** — developer followers, following, public repos, contributions, plus edge features like `isForked` and `isTopContributor`.
3. **Repository popularity** — stars, forks, watchers, contributors, commits.

These three signals are combined in two ways:

- A **linear weighted score** (fast, interpretable baseline).
- A **Random Forest Learning-to-Rank model** trained with positive/negative sampling for a more robust ranker.

## Architecture

```
┌────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
│  developers.csv    │   │  repositories.csv   │   │    edgelist.csv    │
└─────────┬──────────┘   └──────────┬──────────┘   └─────────┬──────────┘
          │                         │                        │
          └────────────┬────────────┴───────────┬────────────┘
                       ▼                        ▼
                ┌──────────────────────────────────────┐
                │   Merge + Feature Engineering        │
                │   (social, popularity, text)         │
                └───────────────┬──────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  TF-IDF +   │      │  Social     │      │  Popularity │
    │  Cosine sim │      │  features   │      │  features   │
    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
           │                    │                    │
           └──────────┬─────────┴──────────┬─────────┘
                      ▼                    ▼
           ┌──────────────────┐   ┌──────────────────────┐
           │  Linear score    │   │  Random Forest LTR   │
           │  (0.5/0.3/0.2)   │   │  (pos/neg sampling)  │
           └────────┬─────────┘   └──────────┬───────────┘
                    └────────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │  Top-K Recommendations│
                     └───────────────────────┘
```

## Dataset

The GitHub Bipartite Graph Dataset consists of three CSV files:

| File | Description | Key columns |
|------|-------------|-------------|
| `developers.csv` | Developer profiles and social metrics | `dev_id`, `Followers`, `Following`, `Public Repositories`, `starredRepoCount`, `yearly_contributions`, `Bio` |
| `repositories.csv` | Repository metadata and content | `repo_id`, `repo_name`, `description`, `readme`, `topics`, `languages`, `stargazers_count`, `forks_count`, `watching`, `contributors_count`, `commits_count` |
| `edgelist.csv` | Developer–repository interactions | `dev_id`, `repo_id`, `isForked`, `isTopContributor` |

> The dataset is not committed to this repository. Place the three CSV files in the `data/` folder before running the notebook or scripts. See [`data/README.md`](data/README.md).

## Project Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md            # How to obtain and place the dataset
├── notebooks/
│   └── recommender_system.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            # Paths, weights, random seeds
│   ├── data_loader.py       # Load + merge the three CSVs
│   ├── features.py          # Social / popularity / text features
│   ├── similarity.py        # TF-IDF + cosine similarity
│   ├── linear_model.py      # Weighted linear scoring
│   ├── ltr_model.py         # Random Forest Learning-to-Rank
│   ├── sampling.py          # Positive / negative sample construction
│   ├── recommend.py         # Top-K recommendation function
│   └── main.py              # Orchestrates the full pipeline
├── reports/
│   └── Recommendation_Systems_Report.pdf
└── docs/
    └── concepts.md          # Short write-up of the 11 concepts
```

## Installation

Requires **Python 3.10+**.

```bash
# Clone
git clone https://github.com/<your-username>/github-hybrid-recommender.git
cd github-hybrid-recommender

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Put the data in place

```
data/
├── developers.csv
├── repositories.csv
└── edgelist.csv
```

### 2. Run the full pipeline from the command line

```bash
python -m src.main --top-n 5 --dev-id <some_dev_id>
```

### 3. Or run the notebook

```bash
jupyter notebook notebooks/recommender_system.ipynb
```

### 4. Use the recommender from Python

```python
from src.main import build_pipeline
from src.recommend import recommend_for_developer

data, ranking_data, ltr_model = build_pipeline()

linear_rec, ltr_rec = recommend_for_developer(
    dev_id="some_developer_id",
    data=data,
    ranking_data=ranking_data,
    top_n=5,
)
print(linear_rec)
print(ltr_rec)
```

## Concepts Implemented

| # | Concept | Where |
|---|---------|-------|
| 1 | Data Acquisition & Preparation | `src/data_loader.py` |
| 2 | User–Item Interaction Matrix | `src/features.py` |
| 3 | TF-IDF Vectorization | `src/similarity.py` |
| 4 | Cosine Similarity | `src/similarity.py` |
| 5 | Feature Engineering & Normalization | `src/features.py` |
| 6 | Linear Weighted Combination | `src/linear_model.py` |
| 7 | Positive & Negative Sampling | `src/sampling.py` |
| 8 | Classification & Evaluation Metrics | `src/ltr_model.py` |
| 9 | Feature Importance Analysis | `src/ltr_model.py` |
| 10 | Recommendation Ranking & Scoring | `src/recommend.py` |
| 11 | Ensemble Methods (Random Forest) | `src/ltr_model.py` |

## Results

Random Forest Learning-to-Rank model on balanced positive/negative samples:

| Metric | Training | Testing |
|--------|----------|---------|
| Accuracy | 0.821 | 0.800 |
| Precision | 0.833 | 0.833 |
| Recall | 0.857 | 0.857 |
| F1-Score | 0.845 | 0.845 |
| ROC-AUC | 0.905 | 0.857 |

Feature importance (Random Forest MDI):

1. **Technical / content score** — primary driver (>50%)
2. **Social score** — secondary signal
3. **Popularity score** — supporting signal

Full write-up with math, derivations, and plots is in [`reports/Recommendation_Systems_Report.pdf`](reports/Recommendation_Systems_Report.pdf).

## Roadmap

- [ ] Train/test split with stratification for unbiased evaluation
- [ ] Add NDCG@K, MAP, and MRR as ranking metrics
- [ ] Replace random negatives with **hard negative mining**
- [ ] Swap Random Forest for **LightGBM LambdaRank**
- [ ] Cold-start strategies for new developers and new repositories
- [ ] Simple FastAPI service exposing `/recommend?dev_id=...`

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- Dataset: GitHub Bipartite Graph Dataset (Developers ↔ Repositories)
- Course: Machine Learning & Recommendation Systems, Spring 2026
