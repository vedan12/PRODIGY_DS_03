# PRODIGY_DS_03 — Decision Tree Classifier (Bank Marketing)

**Prodigy InfoTech — Data Science Internship, Task 03**

## Task
Build a decision tree classifier to predict whether a customer will purchase
a product or service based on their demographic and behavioral data. Use a
dataset such as the Bank Marketing dataset from the UCI Machine Learning
Repository.

## Dataset
`bank.csv` — 11,162 clients of a Portuguese bank, from direct marketing
(phone call) campaigns. Target: `deposit` — did the client subscribe to a
term deposit (yes/no)? Features include age, job, marital status,
education, account balance, loan status, contact type, call duration, and
outcome of previous campaigns.

## Approach
1. **Preprocessing**: label-encoded all 9 categorical columns (job, marital,
   education, default, housing, loan, contact, month, poutcome) — decision
   trees split each feature independently on thresholds, so label encoding
   works fine here (no false ordinal assumption like with a linear model).
2. **Split**: 80/20 train/test split, stratified on the target to preserve
   class balance in both sets.
3. **Model**: `DecisionTreeClassifier` with `max_depth=5` — deliberately
   limited depth to avoid overfitting and to keep the tree visualization
   readable.

## Results
| Metric | Score |
|---|---|
| Accuracy | 80.7% |
| Precision | 0.762 |
| Recall | 0.862 |
| F1 Score | 0.809 |

The model correctly identifies the majority of clients in both classes,
with slightly better recall on the "yes" class (catches most actual
subscribers, at the cost of some false positives).

## Key Insights
- **Call `duration` is by far the strongest predictor** (59% of total
  feature importance) — the longer a client stays on the call, the more
  likely they are to subscribe. This makes intuitive sense: an engaged
  conversation signals real interest.
- **`contact` type, `pdays`** (days since last contact), and **`poutcome`**
  (outcome of the previous campaign) are the next most useful signals —
  clients with a successful prior campaign outcome are far more likely to
  subscribe again.
- Demographic features like **age, job, and education contributed very
  little** to the model compared to behavioral/campaign features —
  **how** a client was contacted and engaged mattered more than **who**
  they are.

## Files
- `task3_decision_tree.py` — full preprocessing, training, and evaluation pipeline
- `bank.csv` — dataset used
- `PRODIGY_DS_03.ipynb` — Colab-ready notebook version
- `output/`
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `decision_tree_plot.png` (first 3 levels, for readability)

## How to run
```bash
pip install pandas scikit-learn matplotlib seaborn
python task3_decision_tree.py
```

---
*Part of the Data Science Internship @ Prodigy InfoTech (Oct 2026)*
#ProdigyInfoTech
