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

    # ── Chart 3: K-means++ vs random init — inertia comparison ──────────────
    # Shows why K-means++ almost always wins: lower final inertia, less variance
    random_inertias, pp_inertias = [], []
    N_TRIALS = 8
    for _ in range(N_TRIALS):
        seed = rng_seed = np.random.randint(0, 9999)
        random_inertias.append(
            KMeans(n_clusters=best_k, init="random", n_init=1,
                   random_state=seed).fit(X_s).inertia_
        )
        pp_inertias.append(
            KMeans(n_clusters=best_k, init="k-means++", n_init=1,
                   random_state=seed).fit(X_s).inertia_
        )

    x = np.arange(N_TRIALS)
    w = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - w/2, random_inertias, w, label="Random init",   color="#d73027", alpha=0.8)
    plt.bar(x + w/2, pp_inertias,     w, label="K-means++ init", color="#1a9850", alpha=0.8)
    plt.xticks(x, [f"Run {i+1}" for i in x], fontsize=9)
    plt.ylabel("Final Inertia (lower = better)")
    plt.title(f"K-Means++ vs Random Initialisation (K={best_k})\n"
              "K-means++ spreads seeds intelligently → lower, more stable inertia")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save("km_pp_vs_random.png")
    charts.append("km_pp_vs_random.png")

    # ── Chart 4: Mini-Batch K-Means — speed vs quality tradeoff ─────────────
    from sklearn.cluster import MiniBatchKMeans
    import time

    batch_sizes  = [32, 64, 128, 256, 512, 1024]
    mb_inertias, mb_times, mb_sil = [], [], []

    for bs in batch_sizes:
        t0 = time.perf_counter()
        mb = MiniBatchKMeans(n_clusters=best_k, batch_size=bs,
                             random_state=42, n_init=3)
        mb_labels = mb.fit_predict(X_s)
        mb_times.append(time.perf_counter() - t0)
        mb_inertias.append(mb.inertia_)
        mb_sil.append(silhouette_score(X_s, mb_labels))

    # full K-Means baseline
    t0 = time.perf_counter()
    full_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    full_km.fit(X_s)
    full_time = time.perf_counter() - t0
    full_inertia = full_km.inertia_
    full_sil = silhouette_score(X_s, full_km.labels_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(batch_sizes, mb_inertias, "o-", color="#3d6aa3", label="Mini-Batch")
    ax1.axhline(full_inertia, color="#d73027", linestyle="--", lw=1.5,
                label=f"Full K-Means ({full_inertia:.0f})")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Mini-Batch K-Means — Inertia vs Batch Size")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(batch_sizes, mb_sil, "s-", color="#1a9850", label="Mini-Batch")
    ax2.axhline(full_sil, color="#d73027", linestyle="--", lw=1.5,
                label=f"Full K-Means ({full_sil:.3f})")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Mini-Batch K-Means — Silhouette vs Batch Size")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Mini-Batch K-Means: larger batches approach full K-Means quality\n"
        "but any batch size is faster on big data",
        fontsize=10
    )
    _save("km_minibatch.png")
    charts.append("km_minibatch.png")

    return {
        "title": "K-Means Clustering",
        "metrics": {
            "Dataset": "unsupervised_placement.csv",
            "Samples": len(df),
            "Features used": len(cluster_cols),
            "Best K (silhouette)": best_k,
            "Best Silhouette Score": round(float(max(sil_scores)), 4),
            "Full K-Means Inertia": round(float(full_inertia), 2),
        },
        "charts": charts,
        "description": (
            "Trained on <b>unsupervised_placement.csv</b> — 4 latent archetypes, no placement label. "
            "Chart 1: Elbow method (inertia vs K). "
            "Chart 2: Silhouette score picks the optimal K. "
            "Chart 3: K-means++ vs random init across 8 independent runs — "
            "K-means++ seeds centroids proportional to distance², giving lower and more stable inertia. "
            "Chart 4: Mini-Batch K-Means trades a small quality drop for speed — "
            "useful when the dataset is too large to fit in memory."
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  Hierarchical Clustering
# ══════════════════════════════════════════════════════════════════

def run_hierarchical() -> dict:
    """
    Agglomerative hierarchical clustering with all four linkage methods
    (single, complete, average, Ward) on the unsupervised placement dataset.
    """
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering

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

    # subsample for dendrogram readability (scipy dendrograms blow up with 2000 pts)
    DEND_N = 300
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_s), DEND_N, replace=False)
    X_dend = X_s[idx]

    linkages = ["ward", "complete", "average", "single"]
    link_colors = {"ward": "#1f3a5f", "complete": "#d73027",
                   "average": "#1a9850", "single": "#ff7f00"}
    link_labels = {"ward": "Ward (minimises within-cluster variance)",
                   "complete": "Complete (max pairwise distance)",
                   "average": "Average (mean pairwise distance)",
                   "single": "Single (min pairwise distance — chaining risk)"}

    # ── Chart 1: Four dendrograms side by side ────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, method in zip(axes, linkages):
        Z = linkage(X_dend, method=method)
        dendrogram(Z, ax=ax, no_labels=True,
                   color_threshold=0.7 * max(Z[:, 2]),
                   above_threshold_color="#aaaaaa",
                   leaf_font_size=6)
        ax.set_title(f"{method.capitalize()} linkage\n{link_labels[method]}",
                     fontsize=8, pad=6)
        ax.set_xlabel(f"Students (n={DEND_N} sample)", fontsize=7)
        ax.set_ylabel("Distance" if method == "ward" else "Dissimilarity", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
    plt.suptitle(
        "Hierarchical Clustering — Dendrograms (4 linkage strategies)\n"
        "Horizontal cut = number of clusters; Ward usually gives most balanced trees",
        fontsize=10, y=1.01
    )
    _save("hc_dendrograms.png")
    charts.append("hc_dendrograms.png")

    # ── Chart 2: Silhouette score per linkage × K ─────────────────────────────
    K_range = range(2, 8)
    plt.figure(figsize=(9, 5))
    best_combo = {"sil": -1}

    for method in linkages:
        sil_scores = []
        for k in K_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=method)
            labels = model.fit_predict(X_s)
            s = silhouette_score(X_s, labels)
            sil_scores.append(s)
            if s > best_combo["sil"]:
                best_combo = {"sil": s, "method": method, "k": k}
        plt.plot(list(K_range), sil_scores, "o-",
                 color=link_colors[method], lw=2, label=method.capitalize())

    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette Score")
    plt.title("Hierarchical Clustering — Silhouette Score per Linkage × K\n"
              "(Ward generally dominates on Euclidean-space data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("hc_silhouette_comparison.png")
    charts.append("hc_silhouette_comparison.png")

    # ── Chart 3: Ward cophenetic distances (how tall each merge is) ───────────
    Z_ward = linkage(X_dend, method="ward")
    merge_heights = Z_ward[:, 2]
    last_merges = merge_heights[-20:][::-1]          # top-20 merges

    plt.figure(figsize=(9, 4))
    plt.bar(range(1, len(last_merges) + 1), last_merges,
            color="#1f3a5f", alpha=0.85, edgecolor="white")
    diffs = np.diff(last_merges[::-1])[::-1]
    biggest_gap_idx = int(np.argmax(diffs)) + 1      # +1 → cut above this merge
    suggested_k = biggest_gap_idx + 1
    plt.axvline(biggest_gap_idx + 0.5, color="#d73027", linestyle="--",
                lw=2, label=f"Biggest gap → K={suggested_k}")
    plt.xlabel("Merge step (last 20, 1 = most recent)")
    plt.ylabel("Merge height (Ward distance)")
    plt.title("Ward Linkage — Last 20 Merge Heights\n"
              "Big gap between consecutive merges signals natural number of clusters")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save("hc_ward_merge_heights.png")
    charts.append("hc_ward_merge_heights.png")

    return {
        "title": "Hierarchical Clustering",
        "metrics": {
            "Dataset": "unsupervised_placement.csv",
            "Samples": len(df),
            "Best linkage": best_combo["method"].capitalize(),
            "Best K": best_combo["k"],
            "Best Silhouette": round(best_combo["sil"], 4),
            "Ward gap suggests K": suggested_k,
        },
        "charts": charts,
        "description": (
            "Agglomerative (bottom-up) hierarchical clustering merges the two closest clusters "
            "at every step. <b>Ward</b> minimises within-cluster variance and usually gives the "
            "most balanced trees. <b>Single</b> linkage is vulnerable to chaining — one stray "
            "point can join two large clusters. Chart 1 shows all four dendrograms; "
            "Chart 2 compares silhouette scores across linkages and K values; "
            "Chart 3 reads the Ward merge-height plot — a large gap between consecutive merges "
            "is the dendrogram equivalent of the elbow method."
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  DBSCAN
# ══════════════════════════════════════════════════════════════════

def run_dbscan() -> dict:
    """
    Density-Based Spatial Clustering of Applications with Noise.
    Shows eps/minPts sensitivity, outlier detection, and why DBSCAN
    handles non-convex clusters that K-Means misses.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    charts = []
    df = pd.read_csv(UNSUPERVISED)

    cluster_cols = [
        "CGPA", "AttendancePercent", "AptitudeTestScore",
        "CodingTestScore", "MockInterviewScore", "SoftSkillsRating",
    ]
    cluster_cols = [c for c in cluster_cols if c in df.columns]

    X = df[cluster_cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # ── Chart 1: k-distance plot to guide eps choice ──────────────────────────
    # Standard heuristic: fit k-NN with k = minPts, sort distances, look for elbow
    MIN_PTS = 5
    nbrs = NearestNeighbors(n_neighbors=MIN_PTS).fit(X_s)
    distances, _ = nbrs.kneighbors(X_s)
    k_dist = np.sort(distances[:, -1])[::-1]

    plt.figure(figsize=(8, 4))
    plt.plot(k_dist, color="#1f3a5f", lw=1.5)
    # rough elbow: biggest second-derivative
    smooth = np.convolve(k_dist, np.ones(20)/20, mode="valid")
    d2 = np.diff(np.diff(smooth))
    elbow_idx = int(np.argmax(d2)) + 20
    elbow_eps  = float(k_dist[elbow_idx])
    plt.axvline(elbow_idx, color="#d73027", linestyle="--", lw=1.5,
                label=f"Elbow ≈ ε={elbow_eps:.2f}")
    plt.xlabel("Points sorted by k-distance (descending)")
    plt.ylabel(f"{MIN_PTS}-NN distance")
    plt.title(f"k-Distance Plot (k=minPts={MIN_PTS})\n"
              "The 'elbow' gives a good starting ε for DBSCAN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save("dbscan_kdist.png")
    charts.append("dbscan_kdist.png")

    # ── Chart 2: eps sweep — clusters found, noise points, silhouette ─────────
    eps_values = np.round(np.linspace(0.3, 2.0, 14), 2)
    n_clusters_list, n_noise_list, sil_list = [], [], []

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=MIN_PTS).fit(X_s)
        labels = db.labels_
        nc = len(set(labels)) - (1 if -1 in labels else 0)
        nn = int(np.sum(labels == -1))
        n_clusters_list.append(nc)
        n_noise_list.append(nn)
        if nc >= 2 and nc < len(X_s) - 1:
            mask = labels != -1
            if mask.sum() > nc:
                sil_list.append(silhouette_score(X_s[mask], labels[mask]))
            else:
                sil_list.append(np.nan)
        else:
            sil_list.append(np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(eps_values, n_clusters_list, "o-", color="#3d6aa3", lw=2)
    axes[0].set_ylabel("Number of clusters found")
    axes[0].set_title(f"DBSCAN Parameter Sweep — ε (minPts={MIN_PTS})")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(elbow_eps, color="#d73027", linestyle="--", lw=1,
                    label=f"Elbow ε≈{elbow_eps:.2f}")
    axes[0].legend(fontsize=8)

    axes[1].bar(eps_values, n_noise_list, width=0.1,
                color="#d73027", alpha=0.8, label="Noise points")
    axes[1].set_ylabel("Noise / Outlier count")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    sil_clean = [s if not np.isnan(s) else None for s in sil_list]
    axes[2].plot(eps_values, sil_list, "s-", color="#1a9850", lw=2)
    axes[2].set_ylabel("Silhouette Score\n(core points only)")
    axes[2].set_xlabel("ε (epsilon)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save("dbscan_eps_sweep.png")
    charts.append("dbscan_eps_sweep.png")

    # ── Chart 3: minPts sweep at fixed eps ───────────────────────────────────
    minpts_values = [2, 3, 4, 5, 8, 10, 15, 20]
    mp_clusters, mp_noise = [], []

    for mp in minpts_values:
        db = DBSCAN(eps=elbow_eps, min_samples=mp).fit(X_s)
        labels = db.labels_
        mp_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
        mp_noise.append(int(np.sum(labels == -1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(minpts_values, mp_clusters, "o-", color="#3d6aa3", lw=2)
    ax1.set_xlabel("minPts")
    ax1.set_ylabel("Clusters found")
    ax1.set_title(f"minPts Sweep (ε={elbow_eps:.2f})\nClusters")
    ax1.grid(True, alpha=0.3)

    ax2.bar(minpts_values, mp_noise, color="#d73027", alpha=0.8, width=1.2)
    ax2.set_xlabel("minPts")
    ax2.set_ylabel("Noise / Outlier points")
    ax2.set_title(f"minPts Sweep (ε={elbow_eps:.2f})\nNoise points (naturally identified outliers)")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("DBSCAN — minPts sensitivity\n"
                 "Higher minPts = stricter density requirement → more outliers",
                 fontsize=10)
    _save("dbscan_minpts_sweep.png")
    charts.append("dbscan_minpts_sweep.png")

    # ── Best run ──────────────────────────────────────────────────────────────
    best_db = DBSCAN(eps=elbow_eps, min_samples=MIN_PTS).fit(X_s)
    best_labels = best_db.labels_
    n_clusters_best = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise_best = int(np.sum(best_labels == -1))

    # ── Chart 4: Outlier fraction per original archetype ─────────────────────
    df_tmp = df.copy()
    df_tmp["dbscan_label"] = best_labels
    df_tmp["is_outlier"] = (best_labels == -1).astype(int)

    arch_outlier = df_tmp.groupby("Archetype")["is_outlier"].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    arch_outlier.plot(kind="bar", color=["#d73027","#ff7f00","#3d6aa3","#1a9850"][:len(arch_outlier)],
                      edgecolor="white")
    plt.ylabel("Fraction flagged as outlier / noise")
    plt.xlabel("Student Archetype")
    plt.title(f"DBSCAN (ε={elbow_eps:.2f}, minPts={MIN_PTS}) — Outlier Rate per Archetype\n"
              "DBSCAN naturally surfaces low-density students as noise")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    _save("dbscan_outlier_by_archetype.png")
    charts.append("dbscan_outlier_by_archetype.png")

    return {
        "title": "DBSCAN",
        "metrics": {
            "Dataset": "unsupervised_placement.csv",
            "ε (auto from elbow)": round(elbow_eps, 2),
            "minPts": MIN_PTS,
            "Clusters found": n_clusters_best,
            "Outliers / Noise": n_noise_best,
            "Outlier %": round(n_noise_best / len(df) * 100, 1),
        },
        "charts": charts,
        "description": (
            "DBSCAN groups points that are within ε of each other with at least minPts neighbours "
            "(core points). Points unreachable from any core point are labelled <b>−1 (noise / outlier)</b> — "
            "no assignment forced. "
            "Chart 1: k-distance plot finds the elbow ε automatically. "
            "Chart 2: ε sweep shows how cluster count and noise fraction change. "
            "Chart 3: minPts sweep — stricter density = more outliers. "
            "Chart 4: outlier rate per archetype confirms DBSCAN naturally exposes "
            "sparse / atypical students as noise."
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
