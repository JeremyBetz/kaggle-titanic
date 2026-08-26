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
| 2 — Feature engineering | `Title`, `IsAlone` | 0.8380, 0.8146, 0.8371, 0.8315, 0.8315 | 0.8305 | 0.0084 | 0.77272 |
| 3 — Model comparison | `Title`, `IsAlone`; Gradient Boosting | 0.8547, 0.8483, 0.8258, 0.8315, 0.8371 | 0.8395 | 0.0107 | 0.77511 |

Experiment 2 evaluates `FamilySize`, `IsAlone`, `Title`, cabin availability, cabin deck, and leakage-safe ticket group size individually, then uses conservative forward selection. The final model improves mean local CV accuracy by 0.0336 over baseline. See `notebooks/02_feature_engineering.ipynb` for hypotheses, fold-level results (including negative results), selection details, and submission verification.

## Experiment 3 model comparison

Experiment 3 holds the final Experiment 2 features and folds fixed while comparing conservative, untuned model families.

| Model | Fold accuracies | Mean accuracy | Standard deviation | Delta vs logistic |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | 0.8380, 0.8146, 0.8371, 0.8315, 0.8315 | 0.8305 | 0.0084 | — |
| Random Forest | 0.8268, 0.7978, 0.7809, 0.8258, 0.8202 | 0.8103 | 0.0181 | -0.0202 |
| Histogram Gradient Boosting | 0.8492, 0.8315, 0.7978, 0.8202, 0.8596 | 0.8316 | 0.0217 | +0.0011 |
| Gradient Boosting | 0.8547, 0.8483, 0.8258, 0.8315, 0.8371 | 0.8395 | 0.0107 | +0.0090 |

Gradient boosting is selected as the Experiment 3 candidate: it improves three paired folds, ties one, and loses one. The gain is reasonably consistent but modest, so it supports a cautious experimental Kaggle submission rather than a strong claim of superiority. See `notebooks/03_model_comparison.ipynb` for preprocessing, timings, tradeoffs, selection reasoning, and submission verification.

## Experiment 4 validation robustness

Using the unchanged Experiment 2 features and unchanged model hyperparameters, Experiment 4 compares logistic regression and gradient boosting over 20 random seeds × 5 paired stratified folds. Gradient boosting averages 0.8325 accuracy across all 100 folds versus 0.8269 for logistic regression. Its mean seed-level paired advantage is +0.0056 (median +0.0062), with 17 wins, 0 ties, and 3 losses across seeds. A 20,000-resample paired seed-bootstrap gives a descriptive 95% interval of [+0.0031, +0.0078].

The seed-42 advantage of +0.0090 is larger than the repeated-validation average, so its magnitude is split-sensitive, but the positive direction is reasonably robust. Gradient boosting remains the current champion, with a modest expected advantage rather than a decisive one. No new submission was created. See `notebooks/04_validation_robustness.ipynb` for every fold and seed result, visualizations, bootstrap details, and limitations.
