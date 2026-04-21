# The 11 Concepts

A condensed reference mapping each concept from the report to its implementation in `src/`.

## 1. Data Acquisition & Preparation
- **Where:** `src/data_loader.py`
- Load `developers.csv`, `repositories.csv`, `edgelist.csv`.
- Merge them into a single interaction frame keyed on `dev_id` and `repo_id`.

## 2. User–Item Interaction Matrix
- **Where:** `src/features.py` → `interaction_matrix_stats`
- Pivot the merged frame into a sparse `developers × repositories` matrix.
- Report shape, density, and sparsity — expect high sparsity (≈63% in the sample data).

## 3. TF-IDF Vectorization
- **Where:** `src/similarity.py` → `build_repo_tfidf`, `build_dev_tfidf`
- `TF(t, d) = freq(t, d) / |d|`
- `IDF(t, D) = log(|D| / |{d : t ∈ d}|)`
- `TF-IDF(t, d, D) = TF · IDF`
- Applied to a combined `description + topics + languages` field per repo, and to developer `Bio`.

## 4. Cosine Similarity
- **Where:** `src/similarity.py` → `compute_technical_scores`
- `sim(A, B) = (A · B) / (‖A‖ · ‖B‖)`
- Each developer is represented by the **centroid** of the TF-IDF vectors of the repos they've interacted with. The technical score for an (`dev_id`, `repo_id`) pair is the cosine similarity between that centroid and the repo's TF-IDF vector.

## 5. Feature Engineering & Normalization
- **Where:** `src/features.py` → `engineer_features`, `normalize_scores`
- `social_score = Followers + Following + Public Repositories + starredRepoCount + yearly_contributions`
- `repo_popularity_score = stargazers + forks + watching + contributors + commits`
- `social_score_final = social_score + isForked + isTopContributor`
- All three components are min-max scaled to `[0, 1]`.

## 6. Linear Weighted Combination
- **Where:** `src/linear_model.py` → `add_linear_score`
- `linear_score = 0.5 · technical_norm + 0.3 · social_norm + 0.2 · popularity_norm`
- Weights configurable in `src/config.py`. Must sum to 1.

## 7. Positive & Negative Sampling
- **Where:** `src/sampling.py` → `build_ranking_dataset`
- Positives: all observed `(dev_id, repo_id)` pairs, `label = 1`.
- Negatives: random `(dev_id, repo_id)` pairs that do **not** appear in the interaction set, `label = 0`.
- For negatives, the features default to the developer's average `technical_norm`/`social_norm` and the repo's average `repo_popularity_norm`, so the model learns from realistic but unobserved profiles.
- Ratio controlled by `NEGATIVE_RATIO` in config (default 1:1).

## 8. Classification & Evaluation Metrics
- **Where:** `src/ltr_model.py` → `evaluate_ltr_model`
- Produces the full `classification_report` (precision / recall / F1 per class) plus ROC-AUC.
- Reference numbers from the midterm report:
  - Accuracy ≈ 0.80
  - Precision / Recall / F1 ≈ 0.83 / 0.86 / 0.85
  - ROC-AUC ≈ 0.86

## 9. Feature Importance Analysis
- **Where:** `src/ltr_model.py` → `feature_importance`
- Random Forest MDI (Mean Decrease in Impurity).
- Observed ordering: **technical_norm > social_norm > repo_popularity_norm**.

## 10. Recommendation Ranking & Scoring
- **Where:** `src/recommend.py` → `recommend_for_developer`
- Two parallel rankings per developer:
  1. Sort observed candidates by `linear_score` (hybrid baseline).
  2. Sort ranking-dataset candidates by `ltr_score` (model output).
- Top-K is configurable via the `--top-n` CLI flag.

## 11. Ensemble Methods (Random Forest)
- **Where:** `src/ltr_model.py` → `train_ltr_model`
- `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)`.
- Bootstrap sampling + random feature subsets at each split reduce variance relative to a single decision tree.

---

### Suggested reading order for graders / reviewers

1. `README.md` (high-level overview)
2. `reports/Recommendation_Systems_Report.pdf` (math and results)
3. `src/main.py` (pipeline flow)
4. Individual modules in the order above
