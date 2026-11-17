# ============ Student Placement Prediction - IMPROVED ============
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------- CONFIG ----------------
CSV_PATH = "student_placement_synthetic.csv"
TARGET = "placement_status"
LEAKAGE_COLS = ["salary_package_lpa"]              # placement ke baad pata chalti hai -> feature nahi
ID_COLS = ["student_id", "id", "name", "roll_no"]  # agar ho to hata denge

# ---------------- STEP 1: Load + Clean ----------------
df = pd.read_csv(CSV_PATH)
print("Original shape:", df.shape)

df = df.drop_duplicates().copy()
df = df.dropna(subset=[TARGET]).copy()     # target ko kabhi mean se fill nahi karte
df[TARGET] = df[TARGET].astype(int)
print("After cleaning:", df.shape)
print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

drop_cols = [c for c in LEAKAGE_COLS + ID_COLS if c in df.columns]
X = df.drop(columns=[TARGET] + drop_cols)
y = df[TARGET]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("\nNumeric features    :", num_cols)
print("Categorical features:", cat_cols)

# ---------------- STEP 2: Split (stratified) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("\nTrain:", len(X_train), " Test:", len(X_test))

# ---------------- STEP 3: Preprocessing ----------------
def make_preprocessor(poly=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if poly:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    num_steps.append(("scaler", StandardScaler()))

    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)

# ---------------- STEP 4: Models + grids ----------------
candidates = {
    "Logistic Regression": (
        Pipeline([("pre", make_preprocessor()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Logistic Regression + Poly": (
        Pipeline([("pre", make_preprocessor(poly=True)),
                  ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        Pipeline([("pre", make_preprocessor()),
                  ("clf", RandomForestClassifier(random_state=RANDOM_STATE,
                                                 class_weight="balanced", n_jobs=-1))]),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [4, 8, None],
         "clf__min_samples_leaf": [1, 5]},
    ),
    "Gradient Boosting": (
        Pipeline([("pre", make_preprocessor()),
                  ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
        {"clf__learning_rate": [0.03, 0.1],
         "clf__max_depth": [2, 3, None],
         "clf__max_iter": [100, 300]},
    ),
}

# ---------------- STEP 5: Cross-validation + GridSearch ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted = [], {}

print("\n=== Model comparison (5-fold CV on training data) ===")
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    fitted[name] = gs.best_estimator_
    results.append({"Model": name, "CV ROC-AUC": gs.best_score_})
    print(f"{name:28s} CV ROC-AUC = {gs.best_score_:.4f}   {gs.best_params_}")

results_df = pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False)
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print("\n>>> Best model:", best_name)

# ---------------- STEP 6: Final test evaluation ----------------
def evaluate(model, Xt, yt):
    pred = model.predict(Xt)
    proba = model.predict_proba(Xt)[:, 1]
    return {
        "Accuracy": accuracy_score(yt, pred),
        "Precision": precision_score(yt, pred, zero_division=0),
        "Recall": recall_score(yt, pred, zero_division=0),
        "F1": f1_score(yt, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(yt, proba),
    }, pred

metrics, y_pred = evaluate(best_model, X_test, y_test)
print("\n=== TEST RESULTS:", best_name, "===")
for k, v in metrics.items():
    print(f"{k:10s}: {v:.3f}")

print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(
    y_test, y_pred, target_names=["Not_Placed", "Placed"], zero_division=0))

# ---- Purane model (3 features, plain LR) se comparison ----
old_cols = [c for c in ["cgpa", "coding_skills", "communication_skills"] if c in X.columns]
old_model = Pipeline([
    ("pre", ColumnTransformer([("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())]), old_cols)])),
    ("clf", LogisticRegression())])
old_model.fit(X_train[old_cols], y_train)
old_metrics, _ = evaluate(old_model, X_test[old_cols], y_test)

compare = pd.DataFrame({"Old (3 features, plain LR)": old_metrics,
                        f"New ({best_name})": metrics}).round(3)
print("\n=== OLD vs NEW (test set) ===")
print(compare)

# ---------------- STEP 7: Charts ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Not_Placed", "Placed"], ax=ax[0], colorbar=False)
ax[0].set_title("Confusion Matrix")
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax[1])
ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
ax[1].set_title("ROC Curve")
plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
plt.show()

# ---------------- STEP 8: Model save ----------------
joblib.dump(best_model, "best_placement_model.pkl")
print("Saved model: best_placement_model.pkl")

# ---------------- STEP 9: Naye student ki prediction ----------------
new_student = {"cgpa": 6.70, "coding_skills": 4.6, "communication_skills": 4.3}

row = {}
for c in X.columns:
    if c in new_student:
        row[c] = new_student[c]
    elif c in num_cols:
        row[c] = X_train[c].median()
    else:
        row[c] = X_train[c].mode()[0]
new_df = pd.DataFrame([row])

pred_class = best_model.predict(new_df)[0]
prob = best_model.predict_proba(new_df)[0]
print("\nNew student:", new_student)
print("Predicted Class :", "Placed" if pred_class == 1 else "Not_Placed")
print(f"P(Not_Placed) = {prob[0]:.3f} | P(Placed) = {prob[1]:.3f}")
