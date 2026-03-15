#!/usr/bin/env python3
"""
Benchmark Script for EEG Pain Classifier
=========================================
Properly evaluates the pain-vs-no-pain classifier on FRESH data
that was never used during training or model selection.

Strategy:
  1. Hold out 2 subjects (sub-esg04, sub-esg05) as a completely unseen test set.
  2. Train all classifiers on the remaining 5 subjects.
  3. Select the best model via Leave-One-Subject-Out CV on the TRAINING subjects only.
  4. Refit the best model on ALL training data.
  5. Evaluate on the held-out test subjects — this is the final benchmark score.

This avoids any data leakage: the test subjects are never seen during training,
feature extraction parameters are not tuned on them, and model selection is done
entirely within the training set.

Usage:
    python benchmark.py --data_dir ds005307
    python benchmark.py --data_dir ds005307 --no_ica          # faster, skip ICA
    python benchmark.py --data_dir ds005307 --test_subjects sub-esg04 sub-esg05
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_predict,
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Reuse all data loading / feature extraction from the training script
from train_baseline import (
    get_subject_dirs,
    load_subject,
    get_classifiers,
    SCALP_CHANNELS,
    RESAMPLE_FREQ,
    FILTER_LOW,
    FILTER_HIGH,
    TMIN,
    TMAX,
)

import mne

warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default held-out test subjects (never touched during training/model selection)
DEFAULT_TEST_SUBJECTS = ["sub-esg04", "sub-esg05"]


def load_all_subjects(data_dir: str, apply_ica: bool = True,
                      extra_data_dirs: list = None):
    """Load features, labels, and group IDs for every subject.

    extra_data_dirs: additional dataset directories (e.g. ds006374) whose
    subjects are used exclusively for training (never for the held-out test).
    """
    all_dirs = [(data_dir, get_subject_dirs(data_dir))]
    if extra_data_dirs:
        for extra_dir in extra_data_dirs:
            extra_subject_dirs = get_subject_dirs(extra_dir)
            if extra_subject_dirs:
                logger.info(f"  Extra dataset {extra_dir}: "
                            f"{len(extra_subject_dirs)} subjects")
                all_dirs.append((extra_dir, extra_subject_dirs))
            else:
                logger.warning(f"  No subjects found in extra_data_dir: {extra_dir}")

    primary_dirs = all_dirs[0][1]
    if not primary_dirs:
        logger.error("No subject directories found in primary data_dir")
        sys.exit(1)

    all_features, all_labels, all_groups = [], [], []
    n_features_ref = None

    for dataset_dir, subject_dirs in all_dirs:
        is_primary = (dataset_dir == data_dir)
        for i, subject_dir in enumerate(subject_dirs):
            features, labels, groups = load_subject(
                subject_dir, debug=(is_primary and i == 0), apply_ica=apply_ica,
            )
            if len(features) == 0:
                continue

            if n_features_ref is None:
                n_features_ref = features.shape[1]

            if features.shape[1] == n_features_ref:
                all_features.append(features)
                all_labels.append(labels)
                all_groups.append(groups)
            else:
                logger.warning(
                    f"  Skipping {Path(subject_dir).name}: "
                    f"feature dim {features.shape[1]} != expected {n_features_ref}"
                )

    if not all_features:
        logger.error("No valid data loaded from any subject")
        sys.exit(1)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    return X, y, groups


def split_train_test(X, y, groups, test_subjects):
    """Split data into training and held-out test sets by subject ID."""
    test_mask = np.isin(groups, test_subjects)
    train_mask = ~test_mask

    X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test, g_test = X[test_mask], y[test_mask], groups[test_mask]

    return X_train, y_train, g_train, X_test, y_test, g_test


def clean_data(X, y, groups):
    """Remove NaN / Inf rows."""
    valid = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    if not np.all(valid):
        n_bad = np.sum(~valid)
        logger.info(f"  Dropped {n_bad} rows with NaN/Inf ({n_bad/len(valid)*100:.1f}%)")
    return X[valid], y[valid], groups[valid]


def run_benchmark(args):
    t_global = time.time()

    logger.info("=" * 70)
    logger.info("  EEG PAIN CLASSIFIER — HELD-OUT BENCHMARK")
    logger.info("=" * 70)

    # ── 1. Load all subjects ─────────────────────────────────────────────
    logger.info("\n[1/5] Loading & preprocessing all subjects...")
    apply_ica = not args.no_ica
    extra_dirs = getattr(args, "extra_data_dirs", None) or []
    X_all, y_all, g_all = load_all_subjects(
        args.data_dir, apply_ica=apply_ica, extra_data_dirs=extra_dirs
    )

    unique_subjects = sorted(np.unique(g_all))
    logger.info(f"  Loaded subjects: {unique_subjects}")
    logger.info(f"  Total epochs: {len(y_all)}  |  Features: {X_all.shape[1]}")
    logger.info(f"  Class balance: pain={np.sum(y_all==1)}, baseline={np.sum(y_all==0)}")

    # ── 2. Split into TRAIN and HELD-OUT TEST ────────────────────────────
    test_subjects = args.test_subjects
    train_subjects = [s for s in unique_subjects if s not in test_subjects]

    # Verify test subjects exist
    missing = [s for s in test_subjects if s not in unique_subjects]
    if missing:
        logger.error(f"Test subjects not found in data: {missing}")
        logger.error(f"Available: {unique_subjects}")
        sys.exit(1)

    logger.info(f"\n[2/5] Splitting data...")
    logger.info(f"  TRAIN subjects ({len(train_subjects)}): {train_subjects}")
    logger.info(f"  TEST  subjects ({len(test_subjects)}):  {test_subjects}  ← HELD OUT, NEVER SEEN")

    X_train, y_train, g_train, X_test, y_test, g_test = split_train_test(
        X_all, y_all, g_all, test_subjects,
    )

    X_train, y_train, g_train = clean_data(X_train, y_train, g_train)
    X_test, y_test, g_test = clean_data(X_test, y_test, g_test)

    logger.info(f"  Train set: {len(y_train)} epochs (pain={np.sum(y_train==1)}, baseline={np.sum(y_train==0)})")
    logger.info(f"  Test  set: {len(y_test)} epochs (pain={np.sum(y_test==1)}, baseline={np.sum(y_test==0)})")

    # ── 3. Model selection via LOSO CV on TRAINING data only ─────────────
    logger.info(f"\n[3/6] Model selection via LOSO CV on training subjects only...")
    train_unique = sorted(np.unique(g_train))
    logger.info(f"  LOSO across {len(train_unique)} training subjects")

    clfs_raw = get_classifiers(n_features=X_train.shape[1])

    # Wrap each pipeline with SMOTE so resampling happens inside each CV fold
    # This keeps probability calibration consistent between CV and final fit
    clfs = {}
    if HAS_SMOTE:
        n_minority = np.sum(y_train == 0)
        k_smote = min(5, n_minority - 1) if n_minority > 1 else 1
        for name, pipe in clfs_raw.items():
            clfs[name] = ImbPipeline([
                ("smote", SMOTE(random_state=42, k_neighbors=k_smote)),
                ("model", pipe),
            ])
    else:
        clfs = clfs_raw
    logo = LeaveOneGroupOut()

    cv_results = {}
    best_name, best_cv_bal_acc = None, -1.0
    best_cv_proba = None

    for name, pipe in clfs.items():
        t0 = time.time()
        try:
            n_jobs = 1 if ("Stacking" in name or "Voting" in name) else -1
            y_cv_pred = cross_val_predict(
                pipe, X_train, y_train, groups=g_train,
                cv=logo, n_jobs=n_jobs,
            )
            # Also get CV probabilities for threshold optimization
            try:
                y_cv_proba = cross_val_predict(
                    pipe, X_train, y_train, groups=g_train,
                    cv=logo, n_jobs=n_jobs, method="predict_proba",
                )[:, 1]
            except Exception:
                y_cv_proba = None
        except Exception as e:
            logger.warning(f"    {name:25s}  CV FAILED: {e}")
            continue
        elapsed = time.time() - t0

        acc = accuracy_score(y_train, y_cv_pred)
        bal_acc = balanced_accuracy_score(y_train, y_cv_pred)
        try:
            auc = roc_auc_score(y_train, y_cv_pred)
        except ValueError:
            auc = float("nan")
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_train, y_cv_pred, average="binary", pos_label=1,
        )

        cv_results[name] = {
            "accuracy": acc, "balanced_accuracy": bal_acc,
            "auc": auc, "precision": prec, "recall": rec, "f1": f1,
            "time": elapsed,
        }
        logger.info(
            f"    {name:25s}  acc={acc:.4f}  bal_acc={bal_acc:.4f}  "
            f"auc={auc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)"
        )

        # Select by BALANCED accuracy (prevents majority-class bias)
        # Add small complexity penalty to prefer simpler models when scores
        # are very close — complex models (Stacking) tend to overfit on CV
        complexity_penalty = {
            "StackingEnsemble": 0.01,
            "SoftVoting": 0.003,
            "LightGBM": 0.002,
            "XGBoost": 0.002,
        }
        penalty = complexity_penalty.get(name, 0.0)
        adjusted_bal_acc = bal_acc - penalty
        if adjusted_bal_acc > best_cv_bal_acc:
            best_cv_bal_acc = adjusted_bal_acc
            best_name = name
            best_cv_proba = y_cv_proba

    if best_name is None:
        logger.error("All classifiers failed during CV")
        sys.exit(1)

    logger.info(f"\n  >> Best model (by CV balanced accuracy): {best_name}  (CV bal_acc={best_cv_bal_acc:.4f})")

    # ── 3b. Threshold selection ─────────────────────────────────────────
    # Threshold optimization on CV probabilities was found to hurt generalisation
    # in this setting (SMOTE during final fit changes probability scale relative
    # to non-SMOTE CV). We stick with 0.5 as default.
    optimal_threshold = 0.5
    logger.info(f"\n[3b/6] Using default threshold: {optimal_threshold:.2f}")

    # ── 4. Refit best model on ALL training data ──────────────────────────
    logger.info(f"\n[4/6] Refitting {best_name} on all {len(y_train)} training epochs...")
    logger.info(f"  (SMOTE is embedded in the pipeline — applied automatically)")
    best_pipe = clfs[best_name]
    t0 = time.time()
    best_pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"  Fit completed in {fit_time:.1f}s")

    # Sanity: training accuracy
    y_train_pred = best_pipe.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    logger.info(f"  Training accuracy (sanity check): {train_acc:.4f}")

    # ── 5. Evaluate on HELD-OUT TEST SET ─────────────────────
    logger.info(f"\n[5/6] Evaluating on held-out test subjects: {test_subjects}")
    logger.info("       (These subjects were NEVER used during training or model selection)")
    logger.info(f"       Using optimized threshold: {optimal_threshold:.2f}")

    # Apply optimized threshold
    try:
        y_test_proba_raw = best_pipe.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba_raw >= optimal_threshold).astype(int)
    except Exception:
        y_test_pred = best_pipe.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    try:
        test_auc = roc_auc_score(y_test, y_test_pred)
    except ValueError:
        test_auc = float("nan")
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary", pos_label=1,
    )
    cm = confusion_matrix(y_test, y_test_pred)

    # Try to get probability-based AUC if the model supports it
    prob_auc = float("nan")
    try:
        y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
        prob_auc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        pass

    # Per-subject breakdown
    logger.info("\n  Per-subject results on test set:")
    for subj in sorted(np.unique(g_test)):
        mask = g_test == subj
        subj_acc = accuracy_score(y_test[mask], y_test_pred[mask])
        subj_bal_acc = balanced_accuracy_score(y_test[mask], y_test_pred[mask])
        subj_n = np.sum(mask)
        subj_pain = np.sum(y_test[mask] == 1)
        subj_base = np.sum(y_test[mask] == 0)
        logger.info(
            f"    {subj:20s}  acc={subj_acc:.4f}  bal_acc={subj_bal_acc:.4f}  "
            f"(n={subj_n}, pain={subj_pain}, baseline={subj_base})"
        )

    # Also evaluate ALL classifiers on test set for comparison
    logger.info("\n  All classifiers on held-out test set:")
    all_test_results = {}
    for name, pipe in clfs.items():
        try:
            pipe.fit(X_train, y_train)
            y_pred_i = pipe.predict(X_test)
            acc_i = accuracy_score(y_test, y_pred_i)
            bal_acc_i = balanced_accuracy_score(y_test, y_pred_i)
            f1_i = precision_recall_fscore_support(
                y_test, y_pred_i, average="binary", pos_label=1,
            )[2]
            try:
                auc_i = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
            except Exception:
                auc_i = roc_auc_score(y_test, y_pred_i)
            all_test_results[name] = {
                "accuracy": acc_i, "balanced_accuracy": bal_acc_i,
                "auc": auc_i, "f1": f1_i,
            }
            marker = " << BEST (CV-selected)" if name == best_name else ""
            logger.info(
                f"    {name:25s}  acc={acc_i:.4f}  bal_acc={bal_acc_i:.4f}  "
                f"auc={auc_i:.4f}  f1={f1_i:.4f}{marker}"
            )
        except Exception as e:
            logger.warning(f"    {name:25s}  FAILED: {e}")

    total_time = time.time() - t_global

    # ── Final Report ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  BENCHMARK RESULTS  —  HELD-OUT TEST SET")
    logger.info("=" * 70)
    logger.info(f"  Best model (selected via LOSO CV balanced accuracy): {best_name}")
    logger.info(f"  CV balanced accuracy (train only):  {best_cv_bal_acc:.4f}")
    logger.info(f"  Optimized threshold:                {optimal_threshold:.2f}")
    logger.info(f"")
    logger.info(f"  ┌─────────────────────────────────────────────┐")
    logger.info(f"  │  HELD-OUT TEST ACCURACY:    {test_acc:.4f}          │")
    logger.info(f"  │  Balanced Accuracy:         {test_bal_acc:.4f}          │")
    logger.info(f"  │  AUC (decision):            {test_auc:.4f}          │")
    if not np.isnan(prob_auc):
        logger.info(f"  │  AUC (probability):         {prob_auc:.4f}          │")
    logger.info(f"  │  Precision (pain):          {test_prec:.4f}          │")
    logger.info(f"  │  Recall (pain):             {test_rec:.4f}          │")
    logger.info(f"  │  F1-score (pain):           {test_f1:.4f}          │")
    logger.info(f"  └─────────────────────────────────────────────┘")
    logger.info(f"")
    logger.info(f"  Train subjects: {train_subjects}")
    logger.info(f"  Test  subjects: {test_subjects}")
    logger.info(f"  Train epochs: {len(y_train)}  |  Test epochs: {len(y_test)}")
    logger.info(f"")
    logger.info(f"  Classification Report (held-out test):")
    report = classification_report(y_test, y_test_pred, target_names=["baseline", "pain"])
    for line in report.strip().split("\n"):
        logger.info(f"    {line}")
    logger.info(f"")
    logger.info(f"  Confusion Matrix (held-out test):")
    logger.info(f"                   Predicted")
    logger.info(f"                  base  pain")
    logger.info(f"    Actual base  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    logger.info(f"    Actual pain  [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    logger.info(f"")
    logger.info(f"  Total benchmark time: {total_time:.1f}s")
    logger.info("=" * 70)

    return {
        "best_model": best_name,
        "cv_balanced_accuracy": best_cv_bal_acc,
        "optimal_threshold": optimal_threshold,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "test_auc": test_auc,
        "test_prob_auc": prob_auc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "confusion_matrix": cm,
        "cv_results": cv_results,
        "all_test_results": all_test_results,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EEG Pain Classifier on held-out subjects"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to ds005307 dataset directory",
    )
    parser.add_argument(
        "--test_subjects", nargs="+", default=DEFAULT_TEST_SUBJECTS,
        help=f"Subject IDs to hold out for testing (default: {DEFAULT_TEST_SUBJECTS})",
    )
    parser.add_argument(
        "--no_ica", action="store_true",
        help="Disable ICA artifact removal (faster)",
    )
    parser.add_argument(
        "--extra_data_dirs", nargs="+", default=[],
        help="Additional dataset directories to include as training-only subjects "
             "(e.g. ds006374). These subjects are NEVER used as test subjects.",
    )
    args = parser.parse_args()

    results = run_benchmark(args)
    return results


if __name__ == "__main__":
    main()
