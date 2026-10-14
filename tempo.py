#!/usr/bin/env python3
# =============================================================================
#  TEMPO
#  Transcriptomic Expression Modeling Pipeline for Omics
#  -----------------------------------------------------------------------------
#  Feature selection, differential expression, trajectory clustering and
#  machine-learning modelling for bulk RNA-seq datasets (nominal or time-course).
#
#  Author  : <your name>
#  Version : 1.0
#  License : MIT
# =============================================================================
"""
TEMPO — Transcriptomic Expression Modeling Pipeline for Omics

Reads a raw count matrix (CSV/TSV) plus a label file, detects whether the
design is nominal or time-course, and produces:

  * Differential expression
      - Global ANOVA across all groups
      - Pairwise Welch t-tests (first_last / consecutive / all)
      - Full tables: raw counts (per sample), means, CPM, log2FC, stat,
        p-value, BH-FDR, and (for time-course) Spearman-rho vs. time
  * Machine-learning feature selection
      - 9 classification selectors x 7 classifiers
      - 7 regression selectors x 7 regressors
      - LOOCV or StratifiedKFold / KFold (auto for small n)
  * K-means clustering of gene trajectories across timepoints
  * Volcano, PCA (raw vs. normalized), ROC, AUC bars, heatmaps
    (per comparison), boxplots, feature importance, confusion matrices,
    learning curves, gene-overlap Jaccard
  * Self-contained HTML report + all TSV tables
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import (StratifiedKFold, KFold, LeaveOneOut,
                                     cross_val_predict, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LogisticRegression, Ridge, Lasso,
                                  ElasticNet)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                       mutual_info_classif,
                                       mutual_info_regression, chi2,
                                       RFE, SelectFromModel,
                                       SequentialFeatureSelector,
                                       VarianceThreshold)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc,
                             matthews_corrcoef, confusion_matrix,
                             r2_score, mean_squared_error, mean_absolute_error)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)
PALETTE = sns.color_palette("tab10")

# -----------------------------------------------------------------------------
# Pipeline identity
# -----------------------------------------------------------------------------
PIPELINE_NAME = "TEMPO"
PIPELINE_LONG = "TEMPO — Transcriptomic Expression Modeling Pipeline for Omics"
PIPELINE_VERSION = "1.0"


# =============================================================================
#  Utilities
# =============================================================================
def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")


def bh_fdr(pvals):
    """Benjamini–Hochberg FDR; NaNs passed through as 1.0."""
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def setup_dirs(outdir):
    outdir = Path(outdir)
    tables = outdir / "tables"
    plots = outdir / "plots"
    for d in (outdir, tables, plots):
        d.mkdir(parents=True, exist_ok=True)
    return outdir, tables, plots


# =============================================================================
#  I/O
# =============================================================================
def load_count_table(path, gene_col=None):
    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] Counts file not found: {path}")
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if gene_col is None:
        gene_col = df.columns[0]
        print(f"[INFO] Using first column as gene IDs: '{gene_col}'")
    if gene_col not in df.columns:
        sys.exit(f"[ERROR] Gene column '{gene_col}' not found.")
    df = df.set_index(gene_col)
    df.index.name = "gene"
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def load_labels(path, samples):
    """
    Labels can be provided two ways:
      * a TSV/CSV file (col1 = sample id, col2 = label)
      * 'auto:<suffix1,suffix2,...' to infer from sample-name suffixes
    """
    if path.startswith("auto:"):
        suffixes = [s.strip() for s in path[5:].split(",")]
        labels = []
        for s in samples:
            matched = next((suf.lstrip("_-.") for suf in suffixes
                            if s.endswith(suf)), None)
            if matched is None:
                sys.exit(f"[ERROR] Cannot infer label for '{s}'. "
                         f"Provide a labels file via --labels.")
            labels.append(matched)
        return pd.Series(labels, index=samples, name="label")

    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] Labels file not found: {path}")
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    lab = pd.read_csv(path, sep=sep)
    if lab.shape[1] < 2:
        lab = pd.read_csv(path, sep=sep, header=None)
    lab = lab.iloc[:, :2]
    lab.columns = ["sample", "label"]
    lab = lab.set_index("sample")["label"].astype(str)
    missing = [s for s in samples if s not in lab.index]
    if missing:
        sys.exit(f"[ERROR] Samples missing in labels file: {missing[:5]}...")
    return lab.reindex(samples)


# =============================================================================
#  Design detection
# =============================================================================
TIME_RE = re.compile(r"^([A-Za-z]+)[_\-]?(\d+(?:\.\d+)?)$")


def detect_design(label_series):
    uniq = list(dict.fromkeys(label_series.values))
    times = {}
    for lab in uniq:
        m = TIME_RE.match(str(lab))
        if not m:
            return "nominal", sorted(uniq), None
        times[lab] = float(m.group(2))
    order = [k for k, _ in sorted(times.items(), key=lambda x: x[1])]
    return "timecourse", order, times


# =============================================================================
#  Preprocessing — returns log2(CPM+1), CPM, and raw
# =============================================================================
def preprocess(counts, min_count=5, min_samples=2, log_transform=True):
    raw = counts.copy()
    mask = (counts >= min_count).sum(axis=1) >= min_samples
    filtered = counts.loc[mask]
    print(f"[INFO] Genes retained: {filtered.shape[0]} / {counts.shape[0]}")

    lib = filtered.sum(axis=0).replace(0, np.nan)
    cpm = filtered.divide(lib, axis=1) * 1e6

    if log_transform:
        log_cpm = np.log2(cpm + 1)
    else:
        log_cpm = cpm.copy()

    # samples x genes
    return log_cpm.T, cpm.T, raw.T


# =============================================================================
#  Differential expression  (with full reporting)
# =============================================================================
def global_de(log_df, cpm_df, raw_df, y, class_order, design="nominal",
              time_values=None):
    """
    One-way ANOVA across all classes; per-gene log2FC (first vs last class),
    plus Spearman rho vs. time for time-course designs.
    Table columns:
      gene (index), mean_raw_<class>, count_<sample>, log2FC_first_vs_last,
      stat, pvalue, padj, [spearman_rho, p_time, padj_time]
    """
    n_cls = len(class_order)
    samples = log_df.index
    rows = []
    for g in log_df.columns:
        v_log = log_df[g].values
        groups = [v_log[y == i] for i in range(n_cls)]
        try:
            f, p_anova = stats.f_oneway(*groups)
        except Exception:
            f, p_anova = np.nan, np.nan

        # log2FC: mean CPM first vs. last class, +1 pseudocount
        try:
            cpm0 = cpm_df[g].values[y == 0]
            cpmL = cpm_df[g].values[y == n_cls - 1]
            log2fc = float(np.log2((np.mean(cpmL) + 1) / (np.mean(cpm0) + 1)))
        except Exception:
            log2fc = np.nan

        row = {
            "gene": g,
            "log2FC_first_vs_last": log2fc,
            "stat": float(f) if not np.isnan(f) else np.nan,
            "pvalue": float(p_anova) if not np.isnan(p_anova) else np.nan,
        }
        # mean raw count per class
        for i, c in enumerate(class_order):
            try:
                row[f"mean_raw_{c}"] = float(np.mean(raw_df[g].values[y == i]))
            except Exception:
                row[f"mean_raw_{c}"] = np.nan
        # per-sample raw counts
        for sid in samples:
            row[f"count_{sid}"] = float(raw_df.loc[sid, g])
        # time-course Spearman
        if design == "timecourse" and time_values is not None:
            try:
                rho, p_time = spearmanr(time_values, v_log)
            except Exception:
                rho, p_time = np.nan, np.nan
            row["spearman_rho"] = float(rho) if not np.isnan(rho) else np.nan
            row["p_time"] = float(p_time) if not np.isnan(p_time) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("gene")
    df["padj"] = bh_fdr(df["pvalue"].values)
    if design == "timecourse":
        df["padj_time"] = bh_fdr(df["p_time"].values)

    df = df.sort_values("pvalue")

    # Column ordering: gene index, per-sample counts, class means, stats
    sample_cols = [c for c in df.columns if c.startswith("count_")]
    mean_cols = [c for c in df.columns if c.startswith("mean_raw_")]
    stat_cols = ["log2FC_first_vs_last", "stat", "pvalue", "padj"]
    if design == "timecourse":
        stat_cols += ["spearman_rho", "p_time", "padj_time"]
    df = df[sample_cols + mean_cols + [c for c in stat_cols if c in df.columns]]
    return df


def pairwise_de(log_df, cpm_df, raw_df, y, class_order, pairs):
    """
    Welch t-test between two conditions on log2(CPM+1).
    Full table includes per-sample raw counts, class means, CPM means,
    log2FC (ratio of CPM means +1), t-statistic, p-value, BH-FDR.
    """
    out = {}
    samples = log_df.index
    for (i, j) in pairs:
        c0, c1 = class_order[i], class_order[j]
        name = f"{c1}_vs_{c0}"

        m = np.isin(y, [i, j])
        y_bin = (y[m] == j).astype(int)
        samples_pair = samples[m]

        rows = []
        for g in log_df.columns:
            v_log = log_df[g].values[m]
            v0 = v_log[y_bin == 0]
            v1 = v_log[y_bin == 1]

            v_cpm = cpm_df[g].values[m]
            vc0 = v_cpm[y_bin == 0]
            vc1 = v_cpm[y_bin == 1]

            v_raw = raw_df[g].values[m]
            vr0 = v_raw[y_bin == 0]
            vr1 = v_raw[y_bin == 1]

            try:
                t, p = stats.ttest_ind(v1, v0, equal_var=False)
            except Exception:
                t, p = np.nan, np.nan

            mc0 = float(np.mean(vc0)) if len(vc0) else np.nan
            mc1 = float(np.mean(vc1)) if len(vc1) else np.nan
            log2fc = float(np.log2((mc1 + 1) / (mc0 + 1))) if (
                not np.isnan(mc0) and not np.isnan(mc1)) else np.nan

            row = {
                "gene": g,
                "mean_raw_c0": float(np.mean(vr0)) if len(vr0) else np.nan,
                "mean_raw_c1": float(np.mean(vr1)) if len(vr1) else np.nan,
                "mean_cpm_c0": mc0,
                "mean_cpm_c1": mc1,
                "mean_log2cpm_c0": float(np.mean(v0)) if len(v0) else np.nan,
                "mean_log2cpm_c1": float(np.mean(v1)) if len(v1) else np.nan,
                "log2FC": log2fc,
                "stat": float(t) if not np.isnan(t) else np.nan,
                "pvalue": float(p) if not np.isnan(p) else np.nan,
            }
            for sid in samples_pair:
                row[f"count_{sid}"] = float(raw_df.loc[sid, g])
            rows.append(row)

        df = pd.DataFrame(rows).set_index("gene")
        df["padj"] = bh_fdr(df["pvalue"].values)
        df = df.sort_values("pvalue")

        # rename means to include class names
        df = df.rename(columns={"mean_raw_c0": f"mean_raw_{c0}",
                                "mean_raw_c1": f"mean_raw_{c1}",
                                "mean_cpm_c0": f"mean_cpm_{c0}",
                                "mean_cpm_c1": f"mean_cpm_{c1}",
                                "mean_log2cpm_c0": f"mean_log2cpm_{c0}",
                                "mean_log2cpm_c1": f"mean_log2cpm_{c1}"})

        count_cols = [c for c in df.columns if c.startswith("count_")]
        mean_cols = [c for c in df.columns if c.startswith("mean_")]
        stat_cols = ["log2FC", "stat", "pvalue", "padj"]
        df = df[count_cols + mean_cols + stat_cols]
        out[name] = df
    return out


def build_pair_list(mode, n_classes):
    if mode == "first_last":
        return [(0, n_classes - 1)]
    if mode == "consecutive":
        return [(i, i + 1) for i in range(n_classes - 1)]
    if mode == "all":
        return list(combinations(range(n_classes), 2))
    raise ValueError(mode)


# =============================================================================
#  Feature selectors  (9 classification, 7 regression)
# =============================================================================
def get_feature_selectors_classification(k, rs=42):
    return {
        # --- filter methods ---
        "VarianceThreshold": VarianceThreshold(threshold=0.0),
        "SelectKBest_f_classif": SelectKBest(score_func=f_classif, k=k),
        "SelectKBest_mutual_info": SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=rs),
            k=k),
        "SelectKBest_chi2": SelectKBest(
            score_func=lambda X, y: chi2(np.abs(X), y), k=k),   # NEW
        # --- wrapper ---
        "RFE_LogisticRegression": RFE(
            estimator=LogisticRegression(max_iter=2000, solver="liblinear"),
            n_features_to_select=k, step=0.2),
        "SequentialFeatureSelector_LogReg": SequentialFeatureSelector(   # NEW
            estimator=LogisticRegression(max_iter=2000, solver="liblinear"),
            n_features_to_select=k, direction="forward", cv=3, n_jobs=-1),
        # --- embedded ---
        "SelectFromModel_RandomForest": SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_ExtraTrees": SelectFromModel(
            estimator=ExtraTreesClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_GradientBoosting": SelectFromModel(
            estimator=GradientBoostingClassifier(random_state=rs),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_AdaBoost": SelectFromModel(                 # NEW
            estimator=AdaBoostClassifier(n_estimators=200, random_state=rs),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_LinearSVC": SelectFromModel(
            estimator=LinearSVC(dual="auto", max_iter=5000, random_state=rs),
            max_features=k, threshold=-np.inf),
    }


def get_feature_selectors_regression(k, rs=42):
    return {
        # --- filter ---
        "SelectKBest_f_regression": SelectKBest(score_func=f_regression, k=k),
        "SelectKBest_mutual_info": SelectKBest(
            score_func=lambda X, y: mutual_info_regression(X, y, random_state=rs),
            k=k),
        # --- wrapper ---
        "SequentialFeatureSelector_Ridge": SequentialFeatureSelector(   # NEW
            estimator=Ridge(alpha=1.0),
            n_features_to_select=k, direction="forward", cv=3, n_jobs=-1),
        # --- embedded ---
        "SelectFromModel_RandomForest": SelectFromModel(
            estimator=RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_ExtraTrees": SelectFromModel(
            estimator=ExtraTreesRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_GradientBoosting": SelectFromModel(
            estimator=GradientBoostingRegressor(random_state=rs),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_AdaBoost": SelectFromModel(                    # NEW
            estimator=AdaBoostRegressor(n_estimators=200, random_state=rs),
            max_features=k, threshold=-np.inf),
        "SelectFromModel_Lasso": SelectFromModel(
            estimator=Lasso(alpha=0.01, random_state=rs, max_iter=5000),
            max_features=k, threshold=-np.inf),
    }


# =============================================================================
#  Classifiers & regressors
# =============================================================================
def get_classifiers(rs=42):
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=rs),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=rs),
        "SVM_Linear": Pipeline([("scaler", StandardScaler()),
                                ("clf", LinearSVC(dual="auto", max_iter=5000))]),
        "KNN": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=1))]),
        "GaussianNB": Pipeline([("scaler", StandardScaler()),
                                ("clf", GaussianNB())]),
    }


def get_regressors(rs=42):
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=0.01, max_iter=5000))]),
        "ElasticNet": Pipeline([("scaler", StandardScaler()),
                                ("reg", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
        "RandomForestReg": RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
        "ExtraTreesReg": ExtraTreesRegressor(n_estimators=300, random_state=rs, n_jobs=-1),
        "GradientBoostingReg": GradientBoostingRegressor(random_state=rs),
        "SVR": Pipeline([("scaler", StandardScaler()), ("reg", SVR(C=1.0, kernel="rbf"))]),
    }


# =============================================================================
#  CV builders
# =============================================================================
def make_classification_cv(y, n_samples, mode="auto"):
    if mode == "loocv" or (mode == "auto" and n_samples <= 12):
        return LeaveOneOut(), "LOOCV"
    min_class = int(np.min(np.bincount(y)))
    n_splits = max(2, min(5, min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), \
        f"StratifiedKFold({n_splits})"


def make_regression_cv(n_samples, mode="auto"):
    if mode == "loocv" or (mode == "auto" and n_samples <= 12):
        return LeaveOneOut(), "LOOCV"
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42), \
        f"KFold({min(5, n_samples)})"


# =============================================================================
#  Evaluation
# =============================================================================
def evaluate_classifier(clf, X, y, cv, n_classes):
    model = clone(clf)
    try:
        y_pred = cross_val_predict(model, X, y, cv=cv)
    except Exception as e:
        return None, None, None, str(e)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
        "roc_auc": np.nan,
    }

    roc_pack = None
    if hasattr(model, "predict_proba"):
        try:
            prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y, prob[:, 1])
                roc_pack = ("binary", y, prob[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y, prob, multi_class="ovr", average="weighted")
                roc_pack = ("multiclass", y, prob)
        except Exception:
            pass
    return metrics, y_pred, roc_pack, None


def evaluate_regressor(reg, X, y_time, cv):
    model = clone(reg)
    try:
        y_pred = cross_val_predict(model, X, y_time, cv=cv)
    except Exception as e:
        return None, None, str(e)

    metrics = {
        "r2":  r2_score(y_time, y_pred),
        "mse": mean_squared_error(y_time, y_pred),
        "mae": mean_absolute_error(y_time, y_pred),
    }
    try:
        if len(np.unique(y_time)) > 2:
            metrics["spearman"] = float(spearmanr(y_time, y_pred).correlation)
        else:
            metrics["spearman"] = np.nan
    except Exception:
        metrics["spearman"] = np.nan
    return metrics, y_pred, None


# =============================================================================
#  K-means gene-trajectory clustering
# =============================================================================
def cluster_genes_over_time(log_df, y, class_order, n_clusters=5, rs=42):
    n_cls = len(class_order)
    mean_mat = np.zeros((n_cls, log_df.shape[1]))
    for i in range(n_cls):
        m = y == i
        if m.sum() > 0:
            mean_mat[i] = log_df.values[m].mean(axis=0)
    mean_df = pd.DataFrame(mean_mat, index=class_order, columns=log_df.columns)

    z_df = mean_df.sub(mean_df.mean(axis=0), axis=1).div(
        mean_df.std(axis=0).replace(0, np.nan), axis=1).fillna(0)

    k = max(2, min(n_clusters, log_df.shape[1] // 2, 10))
    km = KMeans(n_clusters=k, random_state=rs, n_init=10)
    labels_km = km.fit_predict(z_df.T.values)

    cluster_df = pd.DataFrame({"gene": log_df.columns, "cluster": labels_km}
                              ).set_index("gene")
    for c in class_order:
        cluster_df[f"mean_{c}"] = mean_df.loc[c].values
    for c in class_order:
        cluster_df[f"z_{c}"] = z_df.loc[c].values
    cluster_df["cluster"] = cluster_df["cluster"].astype(int)
    return cluster_df, mean_df, z_df


# =============================================================================
#  Plotting
# =============================================================================
def plot_metric_heatmap(summary_df, plots_dir, metric="F1", prefix="clf"):
    if summary_df is None or summary_df.empty or metric not in summary_df.columns:
        return
    pivot = summary_df.pivot(index="feature_selection", columns="classifier",
                             values=metric)
    fig, ax = plt.subplots(figsize=(max(6, 0.65 * pivot.shape[1] + 2.5),
                                    max(3.5, 0.45 * pivot.shape[0] + 1.8)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                cbar_kws={"label": metric})
    ax.set_title(f"{prefix.upper()} {metric} — Feature-selection × Model")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, plots_dir / f"{prefix}_heatmap_{metric}.pdf")


def plot_auc_bars(summary_df, plots_dir):
    if summary_df is None or summary_df.empty or summary_df["ROC_AUC"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * summary_df["classifier"].nunique() * 2), 5))
    sns.barplot(data=summary_df, x="classifier", y="ROC_AUC",
                hue="feature_selection", ax=ax, errorbar=None)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC-AUC by feature-selection method")
    plt.xticks(rotation=30, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    savefig(fig, plots_dir / "auc_bars.pdf")


def plot_roc_curves(roc_data, class_order, plots_dir):
    for fs, per_clf in roc_data.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        any_plotted = False
        for ci, (clf_name, pack) in enumerate(per_clf.items()):
            if pack is None:
                continue
            kind, yt, prob = pack
            color = PALETTE[ci % len(PALETTE)]
            if kind == "binary":
                fpr, tpr, _ = roc_curve(yt, prob)
                ax.plot(fpr, tpr, label=f"{clf_name} (AUC={auc(fpr, tpr):.3f})",
                        color=color)
                any_plotted = True
            else:
                Y = label_binarize(yt, classes=np.arange(len(class_order)))
                fpr_all, tpr_all = [], []
                for c in range(len(class_order)):
                    if Y[:, c].sum() > 0:
                        fpr, tpr, _ = roc_curve(Y[:, c], prob[:, c])
                        fpr_all.append(fpr); tpr_all.append(tpr)
                if not fpr_all:
                    continue
                all_fpr = np.unique(np.concatenate(fpr_all))
                mean_tpr = np.zeros_like(all_fpr)
                for f, t in zip(fpr_all, tpr_all):
                    mean_tpr += np.interp(all_fpr, f, t)
                mean_tpr /= len(fpr_all)
                ax.plot(all_fpr, mean_tpr,
                        label=f"{clf_name} (micro-AUC={auc(all_fpr, mean_tpr):.3f})",
                        color=color)
                any_plotted = True
        if not any_plotted:
            plt.close(fig); continue
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {fs}")
        ax.legend(loc="lower right", fontsize=7)
        savefig(fig, plots_dir / f"roc_{re.sub(r'[^A-Za-z0-9]+', '_', fs)}.pdf")


def plot_pca_comparison(raw_df, norm_df, y, class_order, plots_dir, prefix="pca"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (Xdf, title) in zip(axes, [(raw_df, "Raw counts"),
                                       (norm_df, "log2(CPM+1)")]):
        n_use = min(2000, Xdf.shape[1])
        top = Xdf.var(axis=0).nlargest(n_use).index
        Xv = Xdf.loc[:, top].values
        Xs = StandardScaler().fit_transform(Xv)
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(Xs)
        for i, c in enumerate(class_order):
            m = y == i
            ax.scatter(pcs[m, 0], pcs[m, 1], label=str(c), s=90, alpha=0.85,
                       edgecolor="k", linewidth=0.5,
                       color=PALETTE[i % len(PALETTE)])
        for k_i, sid in enumerate(Xdf.index):
            ax.annotate(str(sid), (pcs[k_i, 0], pcs[k_i, 1]),
                        fontsize=6, alpha=0.75, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("PCA — raw vs. normalized", y=1.02)
    savefig(fig, plots_dir / f"{prefix}_raw_vs_normalized.pdf")


def plot_volcano(de_df, plots_dir, log2fc_cut=1.0, padj_cut=0.05,
                 fname="volcano.pdf", title="Volcano plot",
                 log2fc_col="log2FC"):
    if de_df is None or de_df.empty or log2fc_col not in de_df.columns:
        print(f"[WARN] Volcano skipped ({title}).")
        return
    d = de_df.dropna(subset=[log2fc_col]).copy()
    if d.empty:
        return
    d["-log10padj"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = (d["padj"] < padj_cut) & (d[log2fc_col].abs() > log2fc_cut)
    colors = np.where(sig, "#d62728",
             np.where(d["padj"] < padj_cut, "#ff7f0e", "#7f7f7f"))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d[log2fc_col], d["-log10padj"], c=colors, s=16, alpha=0.75,
               edgecolor="none")
    ax.axvline(log2fc_cut, ls="--", color="grey", lw=1)
    ax.axvline(-log2fc_cut, ls="--", color="grey", lw=1)
    ax.axhline(-np.log10(padj_cut), ls="--", color="grey", lw=1)
    for g, row in d.nsmallest(8, "padj").iterrows():
        ax.annotate(str(g), (row[log2fc_col], row["-log10padj"]),
                    fontsize=6, alpha=0.9)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title(title)
    savefig(fig, plots_dir / fname)


def plot_expression_heatmap(X_df, y, class_order, plots_dir, genes=None,
                            top_n=40, title="Top genes"):
    """Z-scored heatmap of selected genes across samples, with class bar."""
    if genes is None:
        genes = list(X_df.columns[:top_n])
    else:
        genes = [g for g in genes if g in X_df.columns][:top_n]
    if not genes:
        print(f"[WARN] Heatmap '{title}' skipped (no genes).")
        return
    sub = X_df[genes].T
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    lut = {i: PALETTE[i % len(PALETTE)] for i in range(len(class_order))}
    col_colors = pd.Series([lut[i] for i in y], index=X_df.index, name="class")

    cg = sns.clustermap(z, cmap="RdBu_r", center=0,
                        row_cluster=True, col_cluster=True,
                        col_colors=col_colors, figsize=(11, 10),
                        xticklabels=True, yticklabels=True,
                        cbar_kws={"label": "z-score"})
    cg.ax_heatmap.set_xlabel("Samples"); cg.ax_heatmap.set_ylabel("Genes")
    cg.fig.suptitle(title, y=1.02)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          color=lut[i], label=str(class_order[i]))
               for i in range(len(class_order))]
    cg.ax_col_dendrogram.legend(handles=handles, loc="center",
                                bbox_to_anchor=(0.5, 0.5), fontsize=8)
    fname = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower() + ".pdf"
    path = plots_dir / fname
    cg.fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(cg.fig)
    print(f"[PLOT] {path}")


def plot_pairwise_heatmap(log_df, y, class_order, pair, de_df, plots_dir,
                          top_n=40, comparison_name=""):
    """Heatmap of top DE genes for a single pairwise comparison."""
    i, j = pair
    c0, c1 = class_order[i], class_order[j]
    mask = np.isin(y, [i, j])
    samples_sel = log_df.index[mask]
    y_sel = y[mask]

    top_genes = [g for g in de_df.index if g in log_df.columns][:top_n]
    if not top_genes:
        print(f"[WARN] No DE genes for {comparison_name}.")
        return

    sub = log_df.loc[samples_sel, top_genes].T
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    lut = {i: PALETTE[0], j: PALETTE[1]}
    col_colors = pd.Series([lut[v] for v in y_sel],
                           index=samples_sel, name="class")

    cg = sns.clustermap(z, cmap="RdBu_r", center=0,
                        row_cluster=True, col_cluster=True,
                        col_colors=col_colors, figsize=(9, 10),
                        xticklabels=True, yticklabels=True,
                        cbar_kws={"label": "z-score"})
    cg.ax_heatmap.set_xlabel("Samples"); cg.ax_heatmap.set_ylabel("Genes")
    cg.fig.suptitle(f"Top {len(top_genes)} DE genes — {c1} vs {c0}", y=1.02)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          color=lut[i], label=str(c0)),
               plt.Line2D([0], [0], marker="s", linestyle="",
                          color=lut[j], label=str(c1))]
    cg.ax_col_dendrogram.legend(handles=handles, loc="center",
                                bbox_to_anchor=(0.5, 0.5), fontsize=8)
    path = plots_dir / f"heatmap_top_de_genes_{c1}_vs_{c0}.pdf"
    cg.fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(cg.fig)
    print(f"[PLOT] {path}")


def plot_feature_importance_rf(X, y, genes, plots_dir, tables_dir,
                               top_n=25, task="classification"):
    model = (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
             if task == "classification"
             else RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=genes).sort_values(ascending=False)
    imp.to_csv(tables_dir / "rf_gene_importances.tsv", sep="\t",
               header=["importance"])
    top = imp.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * top_n + 1)))
    sns.barplot(x=top.values, y=top.index, ax=ax, color=PALETTE[0])
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} genes — RandomForest ({task})")
    savefig(fig, plots_dir / "feature_importance_rf.pdf")


def plot_top_gene_boxplots(log_df, y, class_order, de_df, plots_dir, n=9):
    top_genes = [g for g in de_df.head(n).index if g in log_df.columns]
    if not top_genes:
        return
    ncol = 3
    nrow = int(np.ceil(len(top_genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_2d(axes)
    for ax, g in zip(axes.flat, top_genes):
        plot_df = pd.DataFrame({
            "expr": log_df[g].values,
            "class": [str(class_order[i]) for i in y],
        })
        sns.boxplot(data=plot_df, x="class", y="expr", ax=ax,
                    palette="Set2", width=0.6)
        sns.stripplot(data=plot_df, x="class", y="expr", ax=ax,
                      color="k", size=3, alpha=0.6)
        ax.set_title(g, fontsize=9); ax.set_xlabel("")
        ax.set_ylabel("log2(CPM+1)")
    for ax in axes.flat[len(top_genes):]:
        ax.axis("off")
    fig.suptitle(f"Top {len(top_genes)} DE genes", y=1.02)
    savefig(fig, plots_dir / "top_gene_boxplots.pdf")


def plot_confusion_matrix(clf, X, y, cv, class_order, plots_dir, tables_dir,
                          fs_name, clf_name):
    y_pred = cross_val_predict(clone(clf), X, y, cv=cv)
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(class_order)))
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
    cm_df.to_csv(tables_dir / f"confusion_{fs_name}__{clf_name}.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False, square=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{fs_name}\n{clf_name}", fontsize=8)
    savefig(fig, plots_dir / f"confusion_{fs_name}__{clf_name}.pdf")


def plot_learning_curve(model, X, y, cv, plots_dir, title, scoring, fname):
    n_samples = len(y)
    train_sizes = np.linspace(0.4, 1.0, min(6, max(3, n_samples - 1)))
    try:
        sizes, tr, te = learning_curve(clone(model), X, y, cv=cv,
                                       scoring=scoring, train_sizes=train_sizes,
                                       n_jobs=-1)
    except Exception as e:
        print(f"[WARN] Learning curve skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sizes, tr.mean(axis=1), "o-", label="Train")
    ax.fill_between(sizes, tr.mean(axis=1) - tr.std(axis=1),
                    tr.mean(axis=1) + tr.std(axis=1), alpha=0.15)
    ax.plot(sizes, te.mean(axis=1), "s-", label="CV")
    ax.fill_between(sizes, te.mean(axis=1) - te.std(axis=1),
                    te.mean(axis=1) + te.std(axis=1), alpha=0.15)
    ax.set_xlabel("Training samples"); ax.set_ylabel(scoring)
    ax.set_title(title); ax.legend()
    savefig(fig, plots_dir / fname)


def plot_gene_overlap(selected_dict, plots_dir, tables_dir,
                      fname="gene_overlap_jaccard.pdf"):
    methods = list(selected_dict.keys())
    if len(methods) < 2:
        return
    J = np.zeros((len(methods), len(methods)))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            sa, sb = set(selected_dict[a]), set(selected_dict[b])
            J[i, j] = len(sa & sb) / max(1, len(sa | sb))
    df = pd.DataFrame(J, index=methods, columns=methods)
    df.to_csv(tables_dir / "gene_overlap_jaccard.tsv", sep="\t",
              float_format="%.4f")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(methods) + 3),) * 2)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Gene-set overlap (Jaccard)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    savefig(fig, plots_dir / fname)


def plot_cv_boxplots(folds_df, plots_dir, metric, prefix="clf"):
    if folds_df is None or folds_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * folds_df["classifier"].nunique() * 2), 5))
    sns.boxplot(data=folds_df, x="classifier", y=metric,
                hue="feature_selection", ax=ax, fliersize=2)
    ax.set_title(f"CV {metric.upper()} distribution")
    plt.xticks(rotation=30, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    savefig(fig, plots_dir / f"{prefix}_cv_boxplot_{metric}.pdf")


def plot_kmeans_clusters(z_df, cluster_df, plots_dir, k):
    class_order = list(z_df.index)
    x = np.arange(len(class_order))
    ncol = min(k, 5)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for c in range(k):
        ax = axes[c]
        genes_c = cluster_df[cluster_df["cluster"] == c].index
        if len(genes_c) == 0:
            ax.set_visible(False); continue
        zc = z_df[genes_c]
        for g in genes_c:
            ax.plot(x, zc[g].values, color="grey", alpha=0.35, lw=0.7)
        ax.plot(x, zc.mean(axis=1).values, color="crimson", lw=2.4)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(class_order, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Cluster {c} (n={len(genes_c)})", fontsize=9)
        if c % ncol == 0:
            ax.set_ylabel("z-score")
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(f"K-means gene trajectories (k={k})", y=1.02)
    savefig(fig, plots_dir / f"kmeans_clusters_k{k}.pdf")


# =============================================================================
#  HTML report
# =============================================================================
def write_html_report(outdir, plots_dir, tables_dir, summary_clf, summary_reg,
                      class_counts, design, args, runtime_s):
    def table(df):
        if df is None or df.empty:
            return "<p><i>(no data)</i></p>"
        cols = list(df.columns)
        rows = "".join(
            "<tr>" + "".join(
                f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                for v in r) + "</tr>"
            for r in df.itertuples(index=False))
        head = "".join(f"<th>{c}</th>" for c in cols)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"

    imgs = sorted(p.name for p in plots_dir.glob("*.pdf"))
    plot_html = "".join(
        f'<div class="plot"><h3>{n}</h3>'
        f'<a href="plots/{n}" target="_blank"><embed src="plots/{n}" '
        f'type="application/pdf" width="100%" height="520px"/></a></div>'
        for n in imgs)
    tables = sorted(p.name for p in tables_dir.glob("*.tsv"))
    tables_html = "".join(f'<li><a href="tables/{t}">{t}</a></li>' for t in tables)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{PIPELINE_NAME} Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
h1 {{ color: #1f4e79; }}
h2 {{ color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: center; }}
th {{ background: #1f4e79; color: white; }}
tr:nth-child(even) {{ background: #f7f9fb; }}
.plot {{ margin: 22px 0; }}
code {{ background:#f3f3f3; padding:2px 4px; }}
.meta {{ font-size: 13px; line-height: 1.6; }}
</style></head><body>
<h1>{PIPELINE_LONG}</h1>
<p class="meta">
<b>Version:</b> {PIPELINE_VERSION} &nbsp;|&nbsp;
<b>Run:</b> {datetime.now().isoformat(timespec='seconds')} &nbsp;|&nbsp;
<b>Runtime:</b> {runtime_s:.1f} s<br>
<b>Counts:</b> <code>{args.counts}</code><br>
<b>Labels:</b> <code>{args.labels}</code><br>
<b>Design:</b> {design} &nbsp;|&nbsp;
<b>Classes:</b> {dict(class_counts)} &nbsp;|&nbsp;
<b>Top-k:</b> {args.top_k} &nbsp;|&nbsp;
<b>CV:</b> {args.cv_mode} &nbsp;|&nbsp;
<b>Pairwise:</b> {args.pairwise_mode}
</p>

<h2>Classification summary</h2>
{table(summary_clf)}

<h2>Regression on continuous time (time-course)</h2>
{table(summary_reg)}

<h2>Plots</h2>
{plot_html}

<h2>Result tables</h2>
<ul>{tables_html}</ul>
</body></html>
"""
    path = outdir / "report.html"
    path.write_text(html)
    print(f"[REPORT] {path}")


# =============================================================================
#  Main
# =============================================================================
def run_pipeline(args):
    t0 = time.time()
    outdir, tables_dir, plots_dir = setup_dirs(args.outdir)

    # ---------- load ----------
    counts = load_count_table(args.counts, args.gene_col)
    samples = list(counts.columns)
    print(f"[INFO] Counts: {counts.shape[0]} genes × {counts.shape[1]} samples")

    labels = load_labels(args.labels, samples)
    design, class_order, time_map = detect_design(labels)
    print(f"[INFO] Design: {design}")
    print(f"[INFO] Class order: {class_order}")
    if design == "timecourse":
        print(f"[INFO] Time values: {time_map}")

    lab2int = {c: i for i, c in enumerate(class_order)}
    y_all = np.array([lab2int[v] for v in labels.values])
    class_counts = {c: int((labels == c).sum()) for c in class_order}

    # ---------- preprocess ----------
    X_log_df, X_cpm_df, X_raw_df = preprocess(
        counts, min_count=args.min_count, min_samples=args.min_samples,
        log_transform=not args.no_log)

    X_log_df = X_log_df.loc[samples]
    X_cpm_df = X_cpm_df.loc[samples]
    X_raw_df = X_raw_df.loc[samples]

    genes = X_log_df.columns.values
    X = X_log_df.values
    y = y_all

    if design == "timecourse":
        time_values = np.array([time_map[c] for c in class_order])[y]
    else:
        time_values = None

    # ---------- global DE ----------
    print("[INFO] Global DE (ANOVA + Spearman) ...")
    global_de_df = global_de(X_log_df, X_cpm_df, X_raw_df, y, class_order,
                             design=design, time_values=time_values)
    global_de_df.to_csv(tables_dir / "global_DE.tsv", sep="\t",
                        float_format="%.5g")
    print(f"[TABLE] global_DE.tsv")

    # ---------- pairwise DE ----------
    pairs_idx = build_pair_list(args.pairwise_mode, len(class_order))
    print(f"[INFO] Pairwise DE mode: {args.pairwise_mode} "
          f"({len(pairs_idx)} comparison(s))")
    pair_results = pairwise_de(X_log_df, X_cpm_df, X_raw_df, y, class_order,
                               pairs_idx)

    # ensure first-vs-last is present for downstream plots
    first_last_key = f"{class_order[-1]}_vs_{class_order[0]}"
    if first_last_key not in pair_results:
        extra = pairwise_de(X_log_df, X_cpm_df, X_raw_df, y, class_order,
                            [(0, len(class_order) - 1)])
        pair_results.update(extra)

    for name, df_pair in pair_results.items():
        df_pair.to_csv(tables_dir / f"DE_{name}.tsv", sep="\t",
                       float_format="%.5g")
        print(f"[TABLE] DE_{name}.tsv")

        # volcano
        plot_volcano(df_pair, plots_dir,
                     fname=f"volcano_{name}.pdf",
                     title=f"Volcano — {name}")

        # pairwise heatmap
        i, j = next((p for p in pairs_idx
                     if f"{class_order[p[1]]}_vs_{class_order[p[0]]}" == name),
                    (0, len(class_order) - 1))
        plot_pairwise_heatmap(X_log_df, y, class_order, (i, j), df_pair,
                              plots_dir, top_n=args.top_n_heatmap,
                              comparison_name=name)

    # ---------- PCA ----------
    plot_pca_comparison(X_raw_df, X_log_df, y, class_order, plots_dir)

    # ---------- K-means clustering ----------
    print("[INFO] K-means clustering of gene trajectories ...")
    try:
        cluster_df, mean_df, z_df = cluster_genes_over_time(
            X_log_df, y, class_order, n_clusters=args.k_clusters)
        cluster_df.to_csv(tables_dir / f"kmeans_clusters_k{args.k_clusters}.tsv",
                          sep="\t", float_format="%.4f")
        mean_df.T.to_csv(tables_dir / "mean_expression_per_timepoint.tsv",
                         sep="\t", float_format="%.4f")
        print(f"[TABLE] kmeans_clusters_k{args.k_clusters}.tsv")
        plot_kmeans_clusters(z_df, cluster_df, plots_dir,
                             k=cluster_df["cluster"].nunique())
    except Exception as e:
        print(f"[WARN] K-means clustering failed: {e}")

    # ---------- classification ----------
    cv_clf, cv_clf_name = make_classification_cv(y, len(y), args.cv_mode)
    print(f"[INFO] CV (classification): {cv_clf_name}")

    k = min(args.top_k, X.shape[1])
    selectors = get_feature_selectors_classification(k)
    classifiers = get_classifiers()

    summary_rows, folds_rows, selected_dict, roc_data = [], [], {}, {}

    for fs_name, selector in selectors.items():
        print(f"\n=== [CLF] {fs_name} ===")
        try:
            selector.fit(X, y)
        except Exception as e:
            print(f"[WARN] Selector failed: {e}")
            continue
        support = (selector.get_support()
                   if hasattr(selector, "get_support")
                   else np.ones(X.shape[1], dtype=bool))
        if support.sum() == 0:
            continue
        X_sel = X[:, support]
        sel_genes = genes[support]
        selected_dict[fs_name] = list(sel_genes)
        print(f"[INFO] Selected {len(sel_genes)} genes")

        roc_data[fs_name] = {}
        for clf_name, clf in classifiers.items():
            metrics, y_pred, roc_pack, err = evaluate_classifier(
                clf, X_sel, y, cv_clf, len(class_order))
            if metrics is None:
                print(f"  {clf_name}: FAILED ({err})")
                continue
            roc_data[fs_name][clf_name] = roc_pack
            row = {"feature_selection": fs_name, "classifier": clf_name,
                   "n_features": int(support.sum()),
                   **metrics}
            summary_rows.append(row)
            folds_rows.append({"feature_selection": fs_name,
                               "classifier": clf_name,
                               "n_samples": len(y),
                               "cv": cv_clf_name,
                               **metrics})
            print(f"  {clf_name:<22s} F1={metrics['f1']:.3f} "
                  f"AUC={metrics['roc_auc']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("f1", ascending=False).reset_index(drop=True)
    summary_df.rename(columns={
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "roc_auc": "ROC_AUC", "mcc": "MCC"}, inplace=True)
    summary_df.to_csv(tables_dir / "classification_summary.tsv",
                      sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(folds_rows).to_csv(
        tables_dir / "classification_aggregate_scores.tsv",
        sep="\t", index=False, float_format="%.4f")
    if selected_dict:
        pd.DataFrame(
            [{"feature_selection": kk, "gene": g}
             for kk, v in selected_dict.items() for g in v]
        ).to_csv(tables_dir / "selected_genes_classification.tsv",
                 sep="\t", index=False)

    # ---------- regression ----------
    summary_reg = pd.DataFrame()
    if design == "timecourse":
        cv_reg, cv_reg_name = make_regression_cv(len(y), args.cv_mode)
        print(f"[INFO] CV (regression): {cv_reg_name}")

        y_time = time_values.astype(float)
        y_time_norm = (y_time - y_time.min()) / (y_time.max() - y_time.min() + 1e-9)

        sel_reg = get_feature_selectors_regression(k)
        regressors = get_regressors()
        reg_rows, reg_selected = [], {}

        for fs_name, selector in sel_reg.items():
            print(f"\n=== [REG] {fs_name} ===")
            try:
                selector.fit(X, y_time_norm)
            except Exception as e:
                print(f"[WARN] Selector failed: {e}")
                continue
            support = (selector.get_support()
                       if hasattr(selector, "get_support")
                       else np.ones(X.shape[1], dtype=bool))
            if support.sum() == 0:
                continue
            X_sel = X[:, support]
            reg_selected[fs_name] = list(genes[support])

            for reg_name, reg in regressors.items():
                metrics, y_pred, err = evaluate_regressor(reg, X_sel,
                                                         y_time_norm, cv_reg)
                if metrics is None:
                    print(f"  {reg_name}: FAILED ({err})")
                    continue
                reg_rows.append({"feature_selection": fs_name,
                                 "regressor": reg_name,
                                 "n_features": int(support.sum()),
                                 **metrics})
                print(f"  {reg_name:<22s} R²={metrics['r2']:.3f} "
                      f"Spearman={metrics['spearman']:.3f}")

        if reg_rows:
            summary_reg = pd.DataFrame(reg_rows).sort_values("r2", ascending=False)
            summary_reg.to_csv(tables_dir / "regression_summary.tsv",
                               sep="\t", index=False, float_format="%.4f")
            pd.DataFrame(
                [{"feature_selection": kk, "gene": g}
                 for kk, v in reg_selected.items() for g in v]
            ).to_csv(tables_dir / "selected_genes_regression.tsv",
                     sep="\t", index=False)
            if len(reg_selected) >= 2:
                plot_gene_overlap(reg_selected, plots_dir, tables_dir,
                                  fname="gene_overlap_jaccard_regression.pdf")

    # ---------- additional plots ----------
    print("\n[INFO] Generating additional plots ...")
    try:
        if not summary_df.empty:
            plot_metric_heatmap(summary_df, plots_dir, "F1", "clf")
            plot_metric_heatmap(summary_df, plots_dir, "Accuracy", "clf")
            plot_metric_heatmap(summary_df, plots_dir, "MCC", "clf")
            plot_auc_bars(summary_df, plots_dir)
            plot_roc_curves(roc_data, class_order, plots_dir)
            plot_cv_boxplots(pd.DataFrame(folds_rows), plots_dir, "f1", "clf")

        # global DE heatmap
        plot_expression_heatmap(X_log_df, y, class_order, plots_dir,
                                genes=list(global_de_df.head(
                                    args.top_n_heatmap).index),
                                title="heatmap_top_de_genes_global")

        # heatmap of top genes for best FS method
        if selected_dict and not summary_df.empty:
            best_fs = summary_df.iloc[0]["feature_selection"]
            best_genes = [g for g in selected_dict.get(best_fs, [])
                          if g in X_log_df.columns]
            if best_genes:
                plot_expression_heatmap(
                    X_log_df, y, class_order, plots_dir,
                    genes=best_genes[:args.top_n_heatmap],
                    title=f"heatmap_selected_genes_{best_fs}")

        plot_feature_importance_rf(X, y, genes, plots_dir, tables_dir,
                                   task="classification")
        plot_top_gene_boxplots(X_log_df, y, class_order,
                               pair_results[first_last_key], plots_dir, n=9)
        plot_gene_overlap(selected_dict, plots_dir, tables_dir)

        # confusion matrices for top-3 classifiers
        if not summary_df.empty:
            for _, r in summary_df.nlargest(3, "F1").iterrows():
                fs_name, clf_name = r["feature_selection"], r["classifier"]
                if fs_name not in selectors or clf_name not in classifiers:
                    continue
                support = (selectors[fs_name].get_support()
                           if hasattr(selectors[fs_name], "get_support")
                           else np.ones(X.shape[1], dtype=bool))
                plot_confusion_matrix(classifiers[clf_name], X[:, support], y,
                                      cv_clf, class_order, plots_dir, tables_dir,
                                      fs_name, clf_name)

        # learning curve
        if not summary_df.empty:
            best = summary_df.iloc[0]
            fs_name, clf_name = best["feature_selection"], best["classifier"]
            if fs_name in selectors and clf_name in classifiers:
                support = (selectors[fs_name].get_support()
                           if hasattr(selectors[fs_name], "get_support")
                           else np.ones(X.shape[1], dtype=bool))
                plot_learning_curve(
                    classifiers[clf_name], X[:, support], y, cv_clf,
                    plots_dir, f"Learning curve — {fs_name} / {clf_name}",
                    "f1_weighted", "learning_curve_best.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[WARN] Plotting error: {e}")

    # ---------- run metadata ----------
    runtime_s = time.time() - t0
    meta = {
        "pipeline": PIPELINE_NAME,
        "pipeline_long": PIPELINE_LONG,
        "version": PIPELINE_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": runtime_s,
        "arguments": vars(args),
        "design": design,
        "class_order": class_order,
        "class_counts": class_counts,
        "n_genes_after_filter": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "cv_classification": cv_clf_name,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[META] {outdir / 'run_metadata.json'}")

    # ---------- HTML report ----------
    write_html_report(outdir, plots_dir, tables_dir, summary_df, summary_reg,
                      class_counts, design, args, runtime_s)

    print(f"\n[DONE] {PIPELINE_NAME} finished in {runtime_s:.1f} s")
    print(f"  • {outdir}/report.html")
    print(f"  • {outdir}/run_metadata.json")
    print(f"  • {tables_dir}/ (TSV tables)")
    print(f"  • {plots_dir}/ (PDF plots)")
    return summary_df


# =============================================================================
#  CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        prog=PIPELINE_NAME,
        description=f"{PIPELINE_LONG} — feature selection, DE, clustering "
                    f"and ML modelling for RNA-seq datasets.")
    p.add_argument("--counts", required=True,
                   help="Raw count matrix (CSV/TSV), genes x samples")
    p.add_argument("--labels", required=True,
                   help="Labels file (sample<TAB>label), OR "
                        "'auto:_suffix1,_suffix2' to infer from sample names")
    p.add_argument("--gene_col", default=None,
                   help="Name of gene ID column (default: first column)")
    p.add_argument("--outdir", default="tempo_results",
                   help="Output directory (default: tempo_results)")
    p.add_argument("--top_k", type=int, default=30,
                   help="Number of genes selected by each ML method")
    p.add_argument("--top_n_heatmap", type=int, default=40,
                   help="Number of top DE genes per heatmap")
    p.add_argument("--k_clusters", type=int, default=5,
                   help="K for k-means gene trajectory clustering")
    p.add_argument("--min_count", type=int, default=5,
                   help="Minimum count threshold for gene filtering")
    p.add_argument("--min_samples", type=int, default=2,
                   help="Minimum samples passing min_count")
    p.add_argument("--no_log", action="store_true",
                   help="Disable CPM + log2 transformation")
    p.add_argument("--cv_mode", choices=["auto", "loocv", "stratified"],
                   default="auto",
                   help="auto = LOOCV for n<=12, k-fold otherwise")
    p.add_argument("--pairwise_mode",
                   choices=["first_last", "consecutive", "all"],
                   default="consecutive",
                   help="Which pairwise DE comparisons to run/plot")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
