"""
PRODIGY_DS_03
--------------
Task: Build a decision tree classifier to predict whether a customer will
purchase a product or service based on their demographic and behavioral
data. Use a dataset such as the Bank Marketing dataset from the UCI
Machine Learning Repository.

Dataset: bank.csv (11,162 clients) — direct marketing campaign records from
a Portuguese bank. Target: 'deposit' (did the client subscribe to a term
deposit? yes/no).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")

# ===========================================================
# 1. LOAD DATA
# ===========================================================
df = pd.read_csv("bank.csv")
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df["deposit"].value_counts())
print(f"\nClass balance: {df['deposit'].value_counts(normalize=True).to_dict()}")
print("\nMissing values:", df.isnull().sum().sum())

# ===========================================================
# 2. PREPROCESSING
# ===========================================================
# Decision trees (sklearn's implementation) need numeric input, so we
# encode every categorical column. Label encoding is fine here because
# decision trees split on thresholds per-feature independently — they
# don't assume any ordering between encoded categories like a linear
# model would.
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=["object", "str"]).columns.tolist()
print("\nCategorical columns encoded:", categorical_cols)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

X = df_encoded.drop(columns=["deposit"])
y = df_encoded["deposit"]  # 0 = no, 1 = yes (alphabetical encoding)

print(f"\nTarget encoding: {dict(zip(encoders['deposit'].classes_, encoders['deposit'].transform(encoders['deposit'].classes_)))}")

# ===========================================================
# 3. TRAIN/TEST SPLIT
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ===========================================================
# 4. TRAIN THE DECISION TREE
# ===========================================================
# max_depth=5 is set deliberately — an unrestricted tree would memorize
# the training data (overfit) and also be unreadable when visualized.
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ===========================================================
# 5. EVALUATE
# ===========================================================
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=encoders["deposit"].classes_))

# ===========================================================
# 6. CONFUSION MATRIX
# ===========================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoders["deposit"].classes_,
            yticklabels=encoders["deposit"].classes_)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved output/confusion_matrix.png")

# ===========================================================
# 7. FEATURE IMPORTANCE
# ===========================================================
importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(x=importance.values, y=importance.index, hue=importance.index, palette="viridis", legend=False)
plt.title("Feature Importance (Decision Tree)", fontsize=14, fontweight="bold")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
plt.close()
print("Saved output/feature_importance.png")
print("\nTop 5 features:")
print(importance.head())

# ===========================================================
# 8. VISUALIZE THE TREE (top 3 levels for readability)
# ===========================================================
plt.figure(figsize=(20, 10))
plot_tree(
    clf, max_depth=3, feature_names=X.columns, class_names=encoders["deposit"].classes_,
    filled=True, rounded=True, fontsize=9
)
plt.title("Decision Tree (first 3 levels)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/decision_tree_plot.png", dpi=150)
plt.close()
print("Saved output/decision_tree_plot.png")

print("\nDone!")
