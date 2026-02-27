#!/usr/bin/env python3
"""
Model Adaptation Script
Fine-tune a pre-trained pain classifier on your own EEG data
to adapt it to your personal brain patterns.

This implements transfer learning for BCI:
1. Load pre-trained model (from train_baseline.py --export)
2. Record a short calibration session with your headset
3. Fine-tune the model on your data
4. Export the personalised model

Usage:
    # Adapt from a BrainVision recording:
    python adapt_model.py --base_model pain_model.pkl \\
                          --calibration_file my_recording.vhdr \\
                          --output my_adapted_model.pkl

    # Adapt with validation split to check improvement:
    python adapt_model.py --base_model pain_model.pkl \\
                          --calibration_file my_recording.vhdr \\
                          --output my_adapted_model.pkl \\
                          --validate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import psd_array_welch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")

# Must match training constants
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

SCALP_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FCz", "Fz", "Cz", "Pz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "FT9", "FT10", "TP9", "TP10",
]

VERTEX_CHANNELS = ["Cz", "FCz", "Fz", "C3", "C4", "FC1", "FC2", "CP1", "CP2"]

PAIN_KEYWORDS = ["pain", "laser", "nociceptive", "stimulus", "stim"]


def extract_calibration_features(
    raw: mne.io.Raw, events_tsv: str, model_meta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features from calibration recording using the same pipeline
    as training. Returns (X, y).
    """
    sfreq = raw.info["sfreq"]
    tmin = model_meta.get("tmin", -0.5)
    tmax = model_meta.get("tmax", 1.0)
    filter_low = model_meta.get("filter_low", 0.5)
    filter_high = model_meta.get("filter_high", 45.0)
    resample_freq = model_meta.get("resample_freq", 250.0)

    # Preprocess
    raw = raw.copy()
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    scalp = [ch for ch in SCALP_CHANNELS if ch in raw.ch_names]
    if scalp:
        raw.pick(scalp)

    try:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    except Exception:
        pass

    raw.filter(l_freq=filter_low, h_freq=filter_high, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=60.0, fir_design="firwin", verbose=False)
    raw.resample(resample_freq, verbose=False)

    # Load events
    events_df = pd.read_csv(events_tsv, sep="\t")
    sfreq = raw.info["sfreq"]

    if "trial_type" in events_df.columns:
        pain_mask = events_df["trial_type"].str.lower().str.contains(
            "|".join(PAIN_KEYWORDS), na=False,
        )
        pain_df = events_df[pain_mask]
    else:
        pain_df = events_df

    # Build events
    pain_events = []
    for _, row in pain_df.iterrows():
        sample = int(round(row["onset"] * sfreq))
        pain_events.append([sample, 0, 1])
    pain_events = np.array(pain_events, dtype=int)

    # Baseline events (inter-trial midpoints)
    pain_onsets = pain_df["onset"].values
    sorted_onsets = np.sort(pain_onsets)
    baseline_events = []
    epoch_dur = abs(tmin) + tmax
    for i in range(len(sorted_onsets) - 1):
        gap = sorted_onsets[i + 1] - sorted_onsets[i]
        if gap > epoch_dur * 2:
            mid = sorted_onsets[i] + gap / 2
            sample = int(round(mid * sfreq))
            if sample > 0:
                baseline_events.append([sample, 0, 2])

    if len(baseline_events) < len(pain_events) // 2:
        baseline_events = []
        for onset in pain_onsets:
            s = int(round((onset - 2.0) * sfreq))
            if s > 0:
                baseline_events.append([s, 0, 2])

    baseline_events = np.array(baseline_events, dtype=int) if baseline_events else np.empty((0, 3), dtype=int)

    if len(baseline_events) > len(pain_events):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(baseline_events), len(pain_events), replace=False)
        baseline_events = baseline_events[idx]

    all_events = np.vstack([pain_events, baseline_events])
    all_events = all_events[all_events[:, 0].argsort()]

    event_id = {"pain": 1, "baseline": 2}
    epochs = mne.Epochs(
        raw, all_events, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=(tmin, tmin + 0.4),
        preload=True, verbose=False,
    )

    labels = np.array([1 if ev[2] == 1 else 0 for ev in epochs.events])

    # Extract features (simplified version matching train_baseline)
    from live_inference import preprocess_epoch
    X_list = []
    data = epochs.get_data(copy=True)
    for i in range(len(epochs)):
        feats = preprocess_epoch(data[i], sfreq, model_meta)
        X_list.append(feats)

    X = np.vstack(X_list)
    return X, labels


def adapt_model(
    base_model: dict, X_cal: np.ndarray, y_cal: np.ndarray,
    alpha: float = 0.5, validate: bool = False,
) -> dict:
    """
    Adapt the base model using calibration data.

    Strategy: Train a new logistic regression head on a mix of
    base model predictions and calibration features.

    alpha: weight for base model predictions vs calibration-only features
    """
    pipeline = base_model["pipeline"]
    n_expected = base_model.get("n_features")

    # Handle feature dimension mismatch
    if n_expected and X_cal.shape[1] != n_expected:
        logger.warning(
            f"Feature dim mismatch: model expects {n_expected}, got {X_cal.shape[1]}. "
            f"Padding/truncating..."
        )
        if X_cal.shape[1] < n_expected:
            pad = np.zeros((X_cal.shape[0], n_expected - X_cal.shape[1]))
            X_cal = np.hstack([X_cal, pad])
        else:
            X_cal = X_cal[:, :n_expected]

    # Get base model predictions as features
    try:
        base_proba = pipeline.predict_proba(X_cal)
    except Exception:
        base_pred = pipeline.predict(X_cal)
        base_proba = np.column_stack([1 - base_pred, base_pred])

    logger.info(f"Base model accuracy on calibration: {accuracy_score(y_cal, pipeline.predict(X_cal)):.4f}")

    if validate:
        # Cross-validate the adapted model
        skf = StratifiedKFold(n_splits=min(5, min(np.sum(y_cal == 0), np.sum(y_cal == 1))))
        scores = []
        for train_idx, test_idx in skf.split(X_cal, y_cal):
            # Combine base predictions with raw features
            X_aug_train = np.hstack([X_cal[train_idx], base_proba[train_idx]])
            X_aug_test = np.hstack([X_cal[test_idx], base_proba[test_idx]])

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_aug_train)
            X_test_s = scaler.transform(X_aug_test)

            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
            clf.fit(X_train_s, y_cal[train_idx])
            scores.append(accuracy_score(y_cal[test_idx], clf.predict(X_test_s)))

        logger.info(f"Adapted model CV accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Final fit on all calibration data
    X_aug = np.hstack([X_cal, base_proba])
    scaler = StandardScaler()
    X_aug_s = scaler.fit_transform(X_aug)

    adapted_clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    adapted_clf.fit(X_aug_s, y_cal)

    adapted_acc = accuracy_score(y_cal, adapted_clf.predict(X_aug_s))
    logger.info(f"Adapted model training accuracy: {adapted_acc:.4f}")

    # Package adapted model
    adapted_model = {
        **base_model,
        "adapted": True,
        "adapted_clf": adapted_clf,
        "adapted_scaler": scaler,
        "base_pipeline": pipeline,
        "adapted_accuracy": adapted_acc,
        "model_name": f"Adapted_{base_model['model_name']}",
    }

    return adapted_model


def main():
    parser = argparse.ArgumentParser(description="Adapt pain classifier to your brain")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to pre-trained model (.pkl)")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="BrainVision .vhdr calibration recording")
    parser.add_argument("--events", type=str, default=None,
                        help="Events TSV (default: inferred from .vhdr)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for adapted model (.pkl)")
    parser.add_argument("--validate", action="store_true",
                        help="Run cross-validation on calibration data")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Adaptation strength (0=base only, 1=calibration only)")
    args = parser.parse_args()

    t0 = time.time()

    # Load base model
    logger.info("Loading base model...")
    base_model = joblib.load(args.base_model)
    logger.info(f"  Model: {base_model['model_name']} (acc={base_model['accuracy']:.4f})")

    # Load calibration data
    logger.info("Loading calibration recording...")
    raw = mne.io.read_raw_brainvision(args.calibration_file, preload=True, verbose=False)

    events_tsv = args.events or args.calibration_file.replace("_eeg.vhdr", "_events.tsv")
    if not Path(events_tsv).exists():
        logger.error(f"Events file not found: {events_tsv}")
        sys.exit(1)

    logger.info("Extracting calibration features...")
    X_cal, y_cal = extract_calibration_features(raw, events_tsv, base_model)
    logger.info(f"  Calibration data: {len(y_cal)} epochs (pain={np.sum(y_cal == 1)}, baseline={np.sum(y_cal == 0)})")

    # Clean
    valid = ~np.any(np.isnan(X_cal) | np.isinf(X_cal), axis=1)
    X_cal, y_cal = X_cal[valid], y_cal[valid]

    # Adapt
    logger.info("Adapting model...")
    adapted = adapt_model(base_model, X_cal, y_cal, alpha=args.alpha, validate=args.validate)

    # Save
    joblib.dump(adapted, args.output)
    logger.info(f"\nAdapted model saved to {args.output}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
