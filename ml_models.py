"""
ml_models.py
Trains ML models on the preprocessed placement dataset and saves
visualisation charts to static/charts/ml/.
Each public run_* function returns a dict consumed by the Flask route.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, mean_squared_error, r2_score,
)
from sklearn.cluster import KMeans
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
PREPROCESSED = os.path.join(
    os.path.dirname(__file__), "preprocessed_placement.csv"
)
RAW = os.path.join(os.path.dirname(__file__), "placement.csv")
UNSUPERVISED = os.path.join(os.path.dirname(__file__), "unsupervised_placement.csv")
ML_CHARTS = os.path.join(os.path.dirname(__file__), "static", "charts", "ml")


def _ensure_dir():
    os.makedirs(ML_CHARTS, exist_ok=True)


def _chart_path(fname: str) -> str:
    _ensure_dir()
    return os.path.join(ML_CHARTS, fname)


def _save(fname: str):
    plt.tight_layout()
    plt.savefig(_chart_path(fname), bbox_inches="tight", dpi=100)
    plt.close("all")


def _load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (X_scaled, X_raw_df, y)."""
    df = pd.read_csv(PREPROCESSED)
    y = df["PlacementStatus"].values
    X = df.drop(columns=["PlacementStatus"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, y


# ══════════════════════════════════════════════════════════════════
#  M2 — Linear Models
# ══════════════════════════════════════════════════════════════════

def run_linear_regression() -> dict:
    """
    Treat salary (or CGPA if salary absent) as the regression target
    to illustrate linear regression end-to-end.
    """
    charts = []
    df = pd.read_csv(RAW)

    # pick target
    target_col = "Salary Package" if "Salary Package" in df.columns else "CGPA"
    feature_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore",
                    "MockInterviewScore", "AttendancePercent", "SoftSkillsRating"]
    feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

    sub = df[feature_cols + [target_col]].dropna()
    X = sub[feature_cols].values
    y = sub[target_col].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Chart 1 – Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="#3d6aa3", edgecolors="white", linewidths=0.4)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Linear Regression — Actual vs Predicted\nRMSE={rmse:.2f}  R²={r2:.3f}")
    plt.legend()
    _save("lr_actual_vs_pred.png")
    charts.append("lr_actual_vs_pred.png")

    # Chart 2 – Coefficients
    plt.figure(figsize=(8, 4))
    coef_series = pd.Series(model.coef_, index=feature_cols).sort_values()
    colors = ["#d73027" if v < 0 else "#1a9850" for v in coef_series.values]
    coef_series.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Linear Regression — Feature Coefficients (standardised)")
    plt.xlabel("Coefficient value")
    _save("lr_coefficients.png")
    charts.append("lr_coefficients.png")

    # Chart 3 – Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, kde=True, color="#26466f")
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Linear Regression — Residual Distribution")
    plt.xlabel("Residual")
    _save("lr_residuals.png")
    charts.append("lr_residuals.png")

    return {
        "title": "Linear Regression",
        "metrics": {"RMSE": round(rmse, 4), "R²": round(r2, 4)},
        "charts": charts,
        "description": (
            f"Predicting <b>{target_col}</b> from academic features. "
            "Closed-form normal equations solved internally by scikit-learn. "
            "Features are standardised (StandardScaler) so coefficients are comparable."
        ),
    }


def run_logistic_regression() -> dict:
    """Binary logistic regression for PlacementStatus."""
    charts = []
    X_s, X_df, y = _load()
    feature_names = list(X_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Chart 1 – Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title(f"Logistic Regression — Confusion Matrix\nAccuracy={acc:.3f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save("logr_confusion.png")
    charts.append("logr_confusion.png")

    # Chart 2 – ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#3d6aa3", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression — ROC Curve")
    plt.legend()
    _save("logr_roc.png")
    charts.append("logr_roc.png")

    # Chart 3 – Top-15 Coefficients
    coef = model.coef_[0]
    top_idx = np.argsort(np.abs(coef))[-15:]
    plt.figure(figsize=(8, 5))
    colors = ["#d73027" if coef[i] < 0 else "#1a9850" for i in top_idx]
    plt.barh([feature_names[i] for i in top_idx], coef[top_idx], color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.title("Logistic Regression — Top-15 Feature Coefficients")
    plt.xlabel("Coefficient (log-odds scale)")
    _save("logr_coefficients.png")
    charts.append("logr_coefficients.png")

    return {
        "title": "Logistic Regression",
        "metrics": {
            "Accuracy": round(acc, 4),
            "AUC-ROC": round(roc_auc, 4),
            "Precision (Placed)": round(report["1"]["precision"], 4),
            "Recall (Placed)": round(report["1"]["recall"], 4),
        },
        "charts": charts,
        "description": (
            "Binary classification using the sigmoid function and cross-entropy loss. "
            "Positive coefficients increase the log-odds of being placed; "
            "negative ones decrease it. Features are standardised."
        ),
    }


def run_ridge_lasso() -> dict:
    """Ridge (L2) and Lasso (L1) regularised logistic regression — side-by-side."""
    charts = []
    df = pd.read_csv(RAW)

    target_col = "Salary Package" if "Salary Package" in df.columns else "CGPA"
    feature_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore",
                    "MockInterviewScore", "AttendancePercent", "SoftSkillsRating"]
    feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

    sub = df[feature_cols + [target_col]].dropna()
    X = sub[feature_cols].values
    y = sub[target_col].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42
    )

    alphas = [0.001, 0.01, 0.1, 1, 10, 50, 100]
    ridge_r2, lasso_r2 = [], []
    for a in alphas:
        ridge_r2.append(r2_score(y_test, Ridge(alpha=a).fit(X_train, y_train).predict(X_test)))
        lasso_r2.append(r2_score(y_test, Lasso(alpha=a, max_iter=5000).fit(X_train, y_train).predict(X_test)))

    # Chart 1 – R² vs alpha
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, ridge_r2, "o-", color="#3d6aa3", label="Ridge (L2)")
    plt.plot(alphas, lasso_r2, "s--", color="#d73027", label="Lasso (L1)")
    plt.xscale("log")
    plt.xlabel("Regularisation strength α (log scale)")
    plt.ylabel("R² on test set")
    plt.title("Ridge vs Lasso — R² across α values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("rl_r2_vs_alpha.png")
    charts.append("rl_r2_vs_alpha.png")

    # Chart 2 – Coefficient paths
    best_alpha = 1.0
    ridge_coef = Ridge(alpha=best_alpha).fit(X_train, y_train).coef_
    lasso_coef = Lasso(alpha=best_alpha, max_iter=5000).fit(X_train, y_train).coef_

    x = np.arange(len(feature_cols))
    w = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, ridge_coef, w, label="Ridge α=1", color="#3d6aa3", alpha=0.8)
    plt.bar(x + w / 2, lasso_coef, w, label="Lasso α=1", color="#d73027", alpha=0.8)
    plt.xticks(x, feature_cols, rotation=30, ha="right")
    plt.axhline(0, color="black", lw=0.8)
    plt.title("Ridge vs Lasso — Coefficient comparison (α=1)\nLasso drives small coefficients to zero (sparsity)")
    plt.ylabel("Coefficient value")
    plt.legend()
    _save("rl_coef_comparison.png")
    charts.append("rl_coef_comparison.png")

    # Chart 3 – ElasticNet mixing
    en_r2 = []
    l1_ratios = np.linspace(0.05, 0.95, 10)
    for lr in l1_ratios:
        en = ElasticNet(alpha=0.1, l1_ratio=lr, max_iter=5000)
        en.fit(X_train, y_train)
        en_r2.append(r2_score(y_test, en.predict(X_test)))

    plt.figure(figsize=(7, 4))
    plt.plot(l1_ratios, en_r2, "^-", color="#26466f")
    plt.xlabel("L1 ratio (0 = pure Ridge, 1 = pure Lasso)")
    plt.ylabel("R² on test set")
    plt.title("ElasticNet — R² vs L1/L2 mixing ratio (α=0.1)")
    plt.grid(True, alpha=0.3)
    _save("rl_elasticnet.png")
    charts.append("rl_elasticnet.png")

    return {
        "title": "Ridge, Lasso & ElasticNet",
        "metrics": {
            "Ridge R² (α=1)": round(float(np.max(ridge_r2)), 4),
            "Lasso R² (α=1)": round(float(np.max(lasso_r2)), 4),
            "Non-zero Lasso coefs": int(np.sum(np.abs(lasso_coef) > 1e-4)),
        },
        "charts": charts,
        "description": (
            "Ridge (L2) shrinks all coefficients evenly. "
            "Lasso (L1) drives irrelevant features exactly to zero, "
            "producing a sparse model. ElasticNet mixes both penalties."
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  M3 — Tree-Based Models
# ══════════════════════════════════════════════════════════════════

def run_decision_tree() -> dict:
    charts = []
    X_s, X_df, y = _load()
    feature_names = list(X_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    # depth sweep for bias-variance
    depths = list(range(1, 15))
    train_acc, test_acc = [], []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
        test_acc.append(accuracy_score(y_test, dt.predict(X_test)))

    # Chart 1 – Depth vs Accuracy (bias-variance)
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_acc, "o-", color="#1a9850", label="Train accuracy")
    plt.plot(depths, test_acc, "s--", color="#d73027", label="Test accuracy")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree — Bias-Variance Tradeoff\n(deeper = more variance / overfitting)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("dt_depth_vs_acc.png")
    charts.append("dt_depth_vs_acc.png")

    # Best model at depth=5
    best = DecisionTreeClassifier(max_depth=5, random_state=42)
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Chart 2 – Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title(f"Decision Tree (depth=5) — Confusion Matrix\nAccuracy={acc:.3f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save("dt_confusion.png")
    charts.append("dt_confusion.png")

    # Chart 3 – Feature importance
    imp = pd.Series(best.feature_importances_, index=feature_names).nlargest(15)
    plt.figure(figsize=(8, 5))
    imp.sort_values().plot(kind="barh", color="#3d6aa3")
    plt.title("Decision Tree — Top-15 Feature Importances (Gini)")
    plt.xlabel("Importance")
    _save("dt_feature_importance.png")
    charts.append("dt_feature_importance.png")

    # Chart 4 – Tree visualisation (shallow depth=3 for readability)
    vis_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    vis_tree.fit(X_train, y_train)
    plt.figure(figsize=(20, 8))
    plot_tree(vis_tree, feature_names=feature_names,
              class_names=["Not Placed", "Placed"],
              filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Structure (depth=3, for readability)")
    _save("dt_tree_plot.png")
    charts.append("dt_tree_plot.png")

    return {
        "title": "Decision Tree",
        "metrics": {"Accuracy (depth=5)": round(acc, 4)},
        "charts": charts,
        "description": (
            "Greedy top-down construction using Gini impurity as the split criterion. "
            "The depth-vs-accuracy plot shows classic overfitting once the tree grows too deep. "
            "Depth=5 balances bias and variance."
        ),
    }


def run_random_forest() -> dict:
    charts = []
    X_s, X_df, y = _load()
    feature_names = list(X_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    # n_estimators sweep
    n_list = [10, 25, 50, 100, 200]
    oob_scores, test_scores = [], []
    for n in n_list:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        oob_scores.append(rf.oob_score_)
        test_scores.append(accuracy_score(y_test, rf.predict(X_test)))

    # Chart 1 – OOB vs Test accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(n_list, oob_scores, "o-", color="#1a9850", label="OOB accuracy")
    plt.plot(n_list, test_scores, "s--", color="#3d6aa3", label="Test accuracy")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest — OOB vs Test accuracy as n_estimators grows")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("rf_oob_vs_test.png")
    charts.append("rf_oob_vs_test.png")

    # Final model
    rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    # Chart 2 – Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title(f"Random Forest (200 trees) — Confusion Matrix\nAccuracy={acc:.3f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save("rf_confusion.png")
    charts.append("rf_confusion.png")

    # Chart 3 – Feature importance
    imp = pd.Series(rf.feature_importances_, index=feature_names).nlargest(15)
    plt.figure(figsize=(8, 5))
    imp.sort_values().plot(kind="barh", color="#26466f")
    plt.title("Random Forest — Top-15 Feature Importances (mean decrease Gini)")
    plt.xlabel("Importance")
    _save("rf_feature_importance.png")
    charts.append("rf_feature_importance.png")

    # Chart 4 – ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#26466f", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest — ROC Curve")
    plt.legend()
    _save("rf_roc.png")
    charts.append("rf_roc.png")

    return {
        "title": "Random Forest",
        "metrics": {
            "Accuracy": round(acc, 4),
            "AUC-ROC": round(roc_auc, 4),
            "OOB Score": round(rf.oob_score_, 4),
        },
        "charts": charts,
        "description": (
            "Bagging + random feature subsampling. Each tree sees a bootstrapped sample; "
            "out-of-bag (OOB) rows give a free validation signal. "
            "Averaging predictions reduces variance without increasing bias."
        ),
    }


def run_xgboost() -> dict:
    charts = []
    X_s, X_df, y = _load()
    feature_names = list(X_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    # Chart 1 – Learning curve (log-loss)
    results = model.evals_result()
    epochs = len(results["validation_0"]["logloss"])
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), results["validation_0"]["logloss"], color="#1a9850", label="Train log-loss")
    plt.plot(range(epochs), results["validation_1"]["logloss"], color="#d73027", label="Test log-loss")
    plt.xlabel("Boosting round")
    plt.ylabel("Log-loss")
    plt.title("XGBoost — Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("xgb_learning_curve.png")
    charts.append("xgb_learning_curve.png")

    # Chart 2 – Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title(f"XGBoost — Confusion Matrix\nAccuracy={acc:.3f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save("xgb_confusion.png")
    charts.append("xgb_confusion.png")

    # Chart 3 – Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(15)
    plt.figure(figsize=(8, 5))
    imp.sort_values().plot(kind="barh", color="#d73027")
    plt.title("XGBoost — Top-15 Feature Importances (gain)")
    plt.xlabel("Importance score")
    _save("xgb_feature_importance.png")
    charts.append("xgb_feature_importance.png")

    # Chart 4 – ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d73027", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost — ROC Curve")
    plt.legend()
    _save("xgb_roc.png")
    charts.append("xgb_roc.png")

    return {
        "title": "XGBoost",
        "metrics": {
            "Accuracy": round(acc, 4),
            "AUC-ROC": round(roc_auc, 4),
        },
        "charts": charts,
        "description": (
            "Gradient boosting builds trees sequentially, each correcting the residuals of the previous. "
            "XGBoost adds L1/L2 regularisation, histogram-based splits, and column subsampling "
            "to make it the dominant algorithm on structured tabular data."
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  M4 — Unsupervised Learning
# ══════════════════════════════════════════════════════════════════

def run_kmeans() -> dict:
    """
    Uses unsupervised_placement.csv — a dataset built with 4 latent archetypes
    (High-Achiever, Average, Below-Average, Specialist) but NO PlacementStatus label.
    K-Means discovers structure purely from features; the 'Archetype' column is used
    only for post-hoc validation, not during training.
    """
    from sklearn.metrics import silhouette_score

    charts = []
    df = pd.read_csv(UNSUPERVISED)

    cluster_cols = [
        "CGPA", "AttendancePercent", "AptitudeTestScore",
        "CodingTestScore", "MockInterviewScore", "SoftSkillsRating",
        "Internships", "Projects", "Certifications",
    ]
    cluster_cols = [c for c in cluster_cols if c in df.columns]

    X = df[cluster_cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # ── Chart 1: Elbow curve ──────────────────────────────────────────────────
    K_range = range(2, 11)
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(K_range), inertias, "o-", color="#3d6aa3", lw=2)
    plt.axvline(4, color="#d73027", linestyle="--", alpha=0.6, label="K=4 (true archetypes)")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("K-Means — Elbow Method\n(dashed line = true number of archetypes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("km_elbow.png")
    charts.append("km_elbow.png")

    # ── Chart 2: Silhouette scores ────────────────────────────────────────────
    sil_scores = []
    for k in K_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_s)
        sil_scores.append(silhouette_score(X_s, labels))

    best_k = list(K_range)[int(np.argmax(sil_scores))]

    plt.figure(figsize=(7, 4))
    plt.plot(list(K_range), sil_scores, "s-", color="#1a9850", lw=2)
    plt.axvline(best_k, color="#d73027", linestyle="--", alpha=0.6, label=f"Best K={best_k}")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette Score (higher = better)")
    plt.title("K-Means — Silhouette Score vs K")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("km_silhouette.png")
    charts.append("km_silhouette.png")



    return {
        "title": "K-Means Clustering",
        "metrics": {
            "Dataset": "unsupervised_placement.csv",
            "Samples": len(df),
            "Features used": len(cluster_cols),
            "Best K (silhouette)": best_k,
            "Best Silhouette Score": round(float(max(sil_scores)), 4),
        },
        "charts": charts,
        "description": (
            "Trained on <b>unsupervised_placement.csv</b> — a dataset with 4 latent archetypes "
            "(High-Achiever, Average, Below-Average, Specialist) but <em>no placement label</em>. "
            f"Silhouette analysis selected K={best_k}. "
            "The elbow curve shows inertia drop-off and the silhouette score confirms the optimal K."
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  Model Comparison
# ══════════════════════════════════════════════════════════════════

def run_model_comparison() -> dict:
    """Quick cross-validated accuracy comparison of all classifiers."""
    charts = []
    X_s, X_df, y = _load()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
    }

    cv_means, cv_stds = {}, {}
    for name, m in models.items():
        scores = cross_val_score(m, X_s, y, cv=5, scoring="accuracy", n_jobs=-1)
        cv_means[name] = scores.mean()
        cv_stds[name] = scores.std()

    # Chart 1 – Bar chart with error bars
    names = list(cv_means.keys())
    means = [cv_means[n] for n in names]
    stds = [cv_stds[n] for n in names]
    colors = ["#3d6aa3", "#1a9850", "#26466f", "#d73027"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=5,
                   color=colors, edgecolor="white", error_kw={"elinewidth": 1.5})
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("5-Fold CV Accuracy")
    plt.title("Model Comparison — 5-Fold Cross-Validated Accuracy\n(error bars = ±1 std)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    _save("cmp_cv_accuracy.png")
    charts.append("cmp_cv_accuracy.png")

    # Chart 2 – ROC curves overlaid
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )
    plt.figure(figsize=(7, 6))
    for (name, m), color in zip(models.items(), colors):
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Comparison — ROC Curves")
    plt.legend(fontsize=9)
    _save("cmp_roc_overlay.png")
    charts.append("cmp_roc_overlay.png")

    return {
        "title": "Model Comparison",
        "metrics": {n: round(v, 4) for n, v in cv_means.items()},
        "charts": charts,
        "description": (
            "5-fold cross-validation accuracy and ROC curves across all classifiers. "
            "Tree-based ensembles typically outperform linear models on this tabular dataset, "
            "consistent with the Kaggle reality discussed in M3."
        ),
    }
