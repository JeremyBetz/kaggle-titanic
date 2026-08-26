# Kaggle Titanic

Reproducible first baseline for the supplied Kaggle Titanic data. This iteration deliberately uses a simple logistic regression and avoids external data, known-solution feature tricks, and leaderboard-driven optimization.

## Data observations

- `train.csv`: 891 rows × 12 columns. The only column absent from `test.csv` is `Survived`, identifying it as the target.
- `test.csv`: 418 rows × 11 columns.
- `gender_submission.csv`: 418 rows × 2 columns. It establishes the required submission schema and ordering as `PassengerId`, `Survived`.
- Target distribution: 549 non-survivors (61.62%) and 342 survivors (38.38%).
- Missing training values: `Age` 177, `Cabin` 687, and `Embarked` 2. Missing test values: `Age` 86, `Fare` 1, and `Cabin` 327.

## Baseline methodology

Features:

- Numeric: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical: `Pclass`, `Sex`, `Embarked`
- Excluded: `PassengerId` (identifier), plus `Name`, `Ticket`, and `Cabin` (high-cardinality/unstructured fields deferred for a later modeling decision)

All preprocessing is inside a scikit-learn `Pipeline` and `ColumnTransformer`. Numeric values are median-imputed and standardized. Categorical values are most-frequent-imputed and one-hot encoded with unknown categories ignored. The estimator is logistic regression with `max_iter=1000` and random seed 42.

Validation uses 5-fold `StratifiedKFold` with shuffling and random seed 42. Preprocessing is fitted separately inside each training fold, preventing validation leakage.

| Fold | Accuracy |
| ---: | ---: |
| 1 | 0.7821 |
| 2 | 0.8034 |
| 3 | 0.7978 |
| 4 | 0.7809 |
| 5 | 0.8202 |
| **Mean** | **0.7969** |
| **Standard deviation** | **0.0146** |

## Reproduce

From the repository root:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/01_baseline.ipynb \
  --output 01_baseline.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=120
```

The executed notebook writes `submissions/submission_01_logistic.csv` after verifying its column names, row count, ID ordering, integer dtypes, missing values, and allowed prediction values against the supplied test and sample-submission files.

## Experiments

All experiment comparisons use the same five shuffled stratified folds (`random_state=42`), accuracy metric, preprocessing strategy, and logistic-regression model family. Feature selection is based only on local cross-validation.

| Experiment | Engineered features | Fold accuracies | Mean accuracy | Standard deviation | Kaggle public score |
| --- | --- | --- | ---: | ---: | ---: |
| 1 — Baseline | None | 0.7821, 0.8034, 0.7978, 0.7809, 0.8202 | 0.7969 | 0.0146 | 0.76555 |
| 2 — Feature engineering | `Title`, `IsAlone` | 0.8380, 0.8146, 0.8371, 0.8315, 0.8315 | 0.8305 | 0.0084 | Not submitted |

Experiment 2 evaluates `FamilySize`, `IsAlone`, `Title`, cabin availability, cabin deck, and leakage-safe ticket group size individually, then uses conservative forward selection. The final model improves mean local CV accuracy by 0.0336 over baseline. See `notebooks/02_feature_engineering.ipynb` for hypotheses, fold-level results (including negative results), selection details, and submission verification.
