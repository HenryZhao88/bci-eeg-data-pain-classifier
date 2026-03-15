#!/usr/bin/env python3
"""
EEG Pain Classification Pipeline v3
for OpenNeuro ds005307 (Laser-evoked Pain EEG Dataset)

v3 improvements over v2.5 (69.5%):
- Wider epoch window (-0.5 to 1.0s) for pre-stimulus alpha
- Average re-reference after channel selection
- Channel interpolation instead of dropping bad channels
- Relative bandpower (normalises across subjects)
- Pre-stimulus alpha power (known pain perception biomarker)
- N1/N2/P2 latency features
- Vertex-channel GFP (Global Field Power)
- Covariance matrix features (upper triangle)
- Filter-Bank CSP spatial filters
- Amplitude rejection for noisy epochs
- Stacking ensemble with LightGBM + XGBoost + LogReg
- Model export via joblib (--export flag)

Usage:
    python train_baseline.py --data_dir ds005307
    python train_baseline.py --data_dir ds005307 --max_subjects 3
    python train_baseline.py --data_dir ds005307 --no_ica
    python train_baseline.py --data_dir ds005307 --export pain_model.pkl
"""

import argparse
import logging
import os
import sys
import time
import warnings
from glob import glob
from pathlib import Path
from typing import Optional

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert as scipy_hilbert

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Filter-Bank CSP bands (narrower sub-bands for spatial filtering)
CSP_BANDS = {
    "theta":      (4, 8),
    "alpha":      (8, 13),
    "low_beta":   (13, 20),
    "high_beta":  (20, 30),
    "gamma":      (30, 45),
}

FILTER_LOW = 0.5
FILTER_HIGH = 45.0
NOTCH_FREQ = 50.0  # ds005307 PowerLineFrequency = 50 Hz (European dataset)
RESAMPLE_FREQ = 250.0  # Downsample from 10 kHz -> 250 Hz after filtering
TMIN = -0.5   # wider window: 500ms pre-stimulus for alpha baseline
TMAX = 1.0    # wider window: 1000ms post-stimulus for full LEP
BASELINE = (-0.5, -0.1)  # Pre-stimulus baseline correction

# Amplitude rejection threshold (peak-to-peak, in Volts for EEG)
REJECT = dict(eeg=150e-6)  # 150 µV

PAIN_KEYWORDS = ["pain", "laser", "nociceptive", "stimulus", "stim"]
# ds006374 uses "repetition" in trial_type for actual laser deliveries
DS006374_PAIN_KEYWORD = "repetition"
DS006374_SECOND_STIM = "2nd"  # use 2nd stimulus only to avoid overlap with 1st

NON_EEG_CHANNELS = [
    "VEOG", "HEOG", "EOG", "ECG", "Biceps", "Resp", "STI 014", "STI014",
]

# Standard 10-20 scalp EEG channels only (exclude spinal cord electrodes)
SCALP_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FCz", "Fz", "Cz", "Pz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "FT9", "FT10", "TP9", "TP10",
]

# Vertex channels (key for laser-evoked potentials)
VERTEX_CHANNELS = ["Cz", "FCz", "Fz", "C3", "C4", "FC1", "FC2", "CP1", "CP2"]


# ── CSP Implementation ──────────────────────────────────────────────────────
class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns for 2-class EEG classification."""

    def __init__(self, n_components: int = 6):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X shape: (n_trials, n_channels, n_times)"""
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes")
        c0, c1 = classes[0], classes[1]

        # Compute covariance matrices for each class
        cov0 = np.mean(
            [np.cov(X[i]) for i in range(len(X)) if y[i] == c0], axis=0
        )
        cov1 = np.mean(
            [np.cov(X[i]) for i in range(len(X)) if y[i] == c1], axis=0
        )

        # Regularise
        reg = 1e-6 * np.eye(cov0.shape[0])
        cov0 += reg
        cov1 += reg

        # Solve generalised eigenvalue problem
        eigenvalues, eigenvectors = eigh(cov0, cov0 + cov1)
        # Sort: most extreme eigenvalues at both ends
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        n = min(self.n_components // 2, eigenvectors.shape[1] // 2)
        # Take first n and last n components
        sel = np.concatenate([np.arange(n), np.arange(-n, 0)])
        self.filters_ = eigenvectors[:, sel].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data through CSP filters and compute log-variance."""
        projected = np.array([self.filters_ @ epoch for epoch in X])
        # Log-variance features
        features = np.log(np.var(projected, axis=2) + 1e-10)
        return features


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-Bank CSP: apply CSP in multiple frequency bands."""

    def __init__(self, bands: dict, n_components: int = 4, sfreq: float = 250.0):
        self.bands = bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.csps_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X shape: (n_trials, n_channels, n_times)"""
        for band_name, (fmin, fmax) in self.bands.items():
            try:
                X_filt = mne.filter.filter_data(
                    X.astype(np.float64), self.sfreq, fmin, fmax,
                    verbose=False,
                )
                csp = CSP(n_components=self.n_components)
                csp.fit(X_filt, y)
                self.csps_[band_name] = csp
            except Exception:
                pass  # Skip bands that fail
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for band_name, csp in self.csps_.items():
            fmin, fmax = self.bands[band_name]
            X_filt = mne.filter.filter_data(
                X.astype(np.float64), self.sfreq, fmin, fmax,
                verbose=False,
            )
            features.append(csp.transform(X_filt))
        if features:
            return np.hstack(features)
        return np.zeros((X.shape[0], 0))


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_subject_dirs(data_dir: str) -> list[str]:
    """Dynamically find all subject directories in the dataset."""
    data_path = Path(data_dir)
    return sorted(
        str(d) for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-") and d.name != "derivatives"
    )


def get_run_files(subject_dir: str) -> list[str]:
    """Get all BrainVision header files for a subject."""
    return sorted(glob(str(Path(subject_dir) / "eeg" / "*_eeg.vhdr")))


def load_bad_channels(vhdr_path: str) -> list[str]:
    """Load bad channels from the corresponding channels.tsv file."""
    channels_path = vhdr_path.replace("_eeg.vhdr", "_channels.tsv")
    if not os.path.exists(channels_path):
        return []
    try:
        df = pd.read_csv(channels_path, sep="\t")
        if "status" in df.columns:
            return df[df["status"] == "bad"]["name"].tolist()
    except Exception as e:
        logger.warning(f"Could not read channels file {channels_path}: {e}")
    return []


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_raw(
    raw: mne.io.Raw,
    bad_channels: Optional[list[str]] = None,
    apply_ica: bool = True,
) -> mne.io.Raw:
    """
    Preprocess raw EEG:
    1. Pick scalp EEG channels only
    2. Interpolate bad channels (preserve channel count)
    3. Average re-reference
    4. Bandpass filter 0.5-45 Hz
    5. Notch filter 60 Hz
    6. ICA-based EOG artifact removal (optional)
    7. Downsample 10 kHz -> 250 Hz
    """
    raw = raw.copy()

    # Pick EEG channels via MNE type system
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)

    # Drop any remaining non-EEG channels by name
    to_drop = [ch for ch in NON_EEG_CHANNELS if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    # Keep only standard scalp EEG channels (exclude spinal cord electrodes)
    scalp_available = [ch for ch in SCALP_CHANNELS if ch in raw.ch_names]
    if scalp_available:
        raw.pick(scalp_available)
        logger.debug(f"Kept {len(scalp_available)} scalp channels")

    # Interpolate bad channels instead of dropping (preserves channel count)
    if bad_channels:
        existing_bad = [ch for ch in bad_channels if ch in raw.ch_names]
        if existing_bad:
            raw.info["bads"] = existing_bad
            try:
                # Need montage for interpolation
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, on_missing="ignore")
                raw.interpolate_bads(reset_bads=True, verbose=False)
                logger.debug(f"Interpolated bad channels: {existing_bad}")
            except Exception:
                raw.drop_channels(existing_bad)
                logger.debug(f"Dropped bad channels (interpolation failed): {existing_bad}")

    # Average re-reference (standard for ERP analysis)
    try:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    except Exception:
        pass

    # Filter BEFORE downsampling to avoid aliasing
    raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, fir_design="firwin", verbose=False)

    # ICA artifact removal - fit on downsampled copy for speed
    if apply_ica and len(raw.ch_names) >= 15:
        try:
            raw_for_ica = raw.copy().resample(RESAMPLE_FREQ, verbose=False)
            # Crop to at most 120s for ICA fitting speed on long recordings
            if raw_for_ica.times[-1] > 120:
                raw_for_ica.crop(tmax=120.0)
            n_components = min(15, len(raw_for_ica.ch_names) - 1)
            ica = mne.preprocessing.ICA(
                n_components=n_components, method="fastica",
                max_iter=200, random_state=42,
            )
            ica.fit(raw_for_ica, verbose=False)
            # Auto-detect EOG-like components using frontal channels
            eog_proxy = [ch for ch in ["Fp1", "Fp2"] if ch in raw_for_ica.ch_names]
            if eog_proxy:
                eog_indices, _ = ica.find_bads_eog(
                    raw_for_ica, ch_name=eog_proxy, verbose=False,
                )
                if eog_indices:
                    ica.exclude = eog_indices[:2]  # Remove at most 2 components
                    ica.apply(raw, verbose=False)
                    logger.debug(f"ICA removed {len(ica.exclude)} EOG components")
            del raw_for_ica
        except Exception as e:
            logger.debug(f"ICA skipped: {e}")

    # Downsample - huge speed gain for all downstream processing
    raw.resample(RESAMPLE_FREQ, verbose=False)

    return raw


# ── Epoching ─────────────────────────────────────────────────────────────────
def create_epochs(
    raw: mne.io.Raw, events_tsv_path: str, debug: bool = False,
) -> tuple[mne.Epochs, np.ndarray]:
    """
    Create epochs around pain stimuli (label=1) and rest baseline (label=0).

    Baseline strategy (adapted for ds005307's short inter-trial intervals):
    1. Extract rest epochs from the pre-stimulus period (recording start → first stim)
    2. Extract rest epochs from the post-stimulus period (last stim → recording end)
    3. Extract rest epochs from any inter-trial gaps large enough (> epoch_duration + 0.5s)
    4. Balance by down-sampling the majority class
    """
    events_df = pd.read_csv(events_tsv_path, sep="\t")

    if debug:
        logger.info(f"events columns: {list(events_df.columns)}")
        for c in ["trial_type", "event_type", "value", "stim_type", "condition"]:
            if c in events_df.columns:
                logger.info(f"unique {c}: {sorted(events_df[c].dropna().unique())[:50]}")

    sfreq = raw.info["sfreq"]
    rec_duration = raw.times[-1]  # total recording length in seconds
    epoch_duration = abs(TMIN) + TMAX  # 1.5s

    # Select only pain stimulus events
    if "trial_type" in events_df.columns:
        # Auto-detect ds006374 format: trial_type contains "repetition" and "2nd"
        # This dataset uses paired stimuli; we use only the 2nd to avoid epoch overlap
        ds006374_mask = (
            events_df["trial_type"].str.contains(DS006374_SECOND_STIM, na=False)
            & events_df["trial_type"].str.contains(DS006374_PAIN_KEYWORD, na=False)
        )
        if ds006374_mask.any() and not events_df["trial_type"].str.lower().str.contains(
            "|".join(PAIN_KEYWORDS), na=False
        ).any():
            # ds006374-style dataset: "2nd/repetition/..." events are pain
            pain_df = events_df[ds006374_mask]
            if debug:
                logger.info(f"  Auto-detected ds006374 format: "
                            f"{len(pain_df)} '2nd/repetition' events as pain")
        else:
            pain_mask = events_df["trial_type"].str.lower().str.contains(
                "|".join(PAIN_KEYWORDS), na=False,
            )
            pain_df = events_df[pain_mask]
    else:
        logger.warning("No trial_type column - using all events as pain stimuli")
        pain_df = events_df

    if debug:
        logger.info(
            f"Selected {len(pain_df)} pain events out of {len(events_df)} total"
        )

    # Build pain events array
    pain_onsets = pain_df["onset"].values
    pain_events = []
    for onset in pain_onsets:
        sample = int(round(onset * sfreq))
        pain_events.append([sample, 0, 1])
    pain_events = np.array(pain_events, dtype=int)

    sorted_onsets = np.sort(pain_onsets)

    # ── Build baseline events from genuine rest periods ──
    baseline_events = []
    # Use a tighter step for more baseline epochs (allow 25% overlap)
    step = epoch_duration * 0.75  # ~1.125s step

    # (a) Pre-stimulus rest: from TMIN offset to first_onset - safety margin
    first_onset = sorted_onsets[0] if len(sorted_onsets) > 0 else rec_duration
    rest_start = abs(TMIN)  # need at least |TMIN| before the epoch center
    rest_end = first_onset - TMAX - 0.2  # safety margin from first pain epoch
    t = rest_start
    while t + TMAX <= rest_end + TMAX and t < rest_end:
        sample = int(round(t * sfreq))
        baseline_events.append([sample, 0, 2])
        t += step

    # (b) Post-stimulus rest: from last_onset + safety to recording end
    if len(sorted_onsets) > 0:
        last_onset = sorted_onsets[-1]
        rest_start = last_onset + TMAX + 0.5  # well after last pain response
        rest_end = rec_duration - TMAX  # don't exceed recording
        t = rest_start
        while t + TMAX <= rec_duration and t < rest_end:
            sample = int(round(t * sfreq))
            baseline_events.append([sample, 0, 2])
            t += step

    # (c) Inter-trial gaps: extract MULTIPLE baseline epochs per gap
    for i in range(len(sorted_onsets) - 1):
        gap_start = sorted_onsets[i] + TMAX + 0.2  # after previous pain response
        gap_end = sorted_onsets[i + 1] - abs(TMIN) - 0.2  # before next pain epoch
        available = gap_end - gap_start
        if available >= epoch_duration:
            t = gap_start
            while t + TMAX <= sorted_onsets[i + 1] - 0.2 and t < gap_end:
                sample = int(round(t * sfreq))
                if sample > 0:
                    baseline_events.append([sample, 0, 2])
                t += step

    # (d) Pre-stimulus baselines: use period well before each pain stimulus
    #     With rapid ITI (~1.65s), the ~0.15s gap between pain epochs is unusable,
    #     but we can take the pre-stimulus portion of trials where it doesn't
    #     overlap with the PREVIOUS trial's response window.
    #     For trials with enough preceding rest (> 2.5s before), extract baseline
    #     centered 2s before the pain onset.
    for idx_o, onset in enumerate(sorted_onsets):
        center = onset - 2.0  # 2s before the pain stimulus
        # Epoch would span: center + TMIN to center + TMAX = onset-2.5 to onset-1.0
        # Check it doesn't overlap with preceding pain epoch (which ends at prev+TMAX)
        preceding = sorted_onsets[sorted_onsets < onset]
        if len(preceding) > 0:
            prev_end = preceding[-1] + TMAX  # end of previous pain response
            epoch_start = center + TMIN  # start of this baseline epoch
            if epoch_start < prev_end + 0.1:  # would overlap
                continue
        # Also check epoch start is within recording
        if center + TMIN < 0:
            continue
        sample = int(round(center * sfreq))
        if sample > 0:
            baseline_events.append([sample, 0, 2])

    baseline_events = np.array(baseline_events, dtype=int) if baseline_events else np.empty((0, 3), dtype=int)

    if debug:
        logger.info(f"Created {len(baseline_events)} baseline epochs from rest periods")

    # Balance classes: down-sample majority to 1:1
    n_pain = len(pain_events)
    n_base = len(baseline_events)
    rng = np.random.RandomState(42)

    if n_base > n_pain:
        idx = rng.choice(n_base, n_pain, replace=False)
        baseline_events = baseline_events[idx]
    elif n_base > 0 and n_pain > n_base * 2:
        # Down-sample pain to at most 2× baseline (tighter than before)
        max_pain = n_base * 2
        idx = rng.choice(n_pain, max_pain, replace=False)
        pain_events = pain_events[idx]
        logger.info(f"  Down-sampled pain epochs {n_pain} -> {max_pain} (2:1 ratio)")

    if len(baseline_events) == 0:
        logger.warning("No baseline events could be created from rest periods")
        return mne.Epochs(raw, pain_events, event_id={"pain": 1},
                          tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                          preload=True, verbose=False), np.ones(len(pain_events))

    all_events = np.vstack([pain_events, baseline_events])
    all_events = all_events[all_events[:, 0].argsort()]

    event_id = {"pain": 1, "baseline": 2}

    try:
        epochs = mne.Epochs(
            raw, all_events, event_id=event_id,
            tmin=TMIN, tmax=TMAX, baseline=BASELINE,
            preload=True, verbose=False,
            reject=REJECT, reject_by_annotation=True,
        )
    except Exception:
        try:
            epochs = mne.Epochs(
                raw, all_events, event_id=event_id,
                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                preload=True, verbose=False, reject=REJECT,
            )
        except Exception:
            epochs = mne.Epochs(
                raw, all_events, event_id=event_id,
                tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                preload=True, verbose=False,
            )

    labels = np.array([1 if ev[2] == 1 else 0 for ev in epochs.events])
    return epochs, labels


# ── Feature Extraction ───────────────────────────────────────────────────────
def extract_features(epochs: mne.Epochs) -> np.ndarray:
    """
    Extract comprehensive feature set from epochs:
    1. Absolute + relative bandpower in 5 bands (vectorized PSD)
    2. Band-power ratios (alpha/beta, theta/beta, theta/alpha)
    3. Pre-stimulus alpha power (pain anticipation biomarker)
    4. Temporal statistics: mean, var, skew, kurtosis, peak-to-peak, RMS
    5. ERP windows: N1 (50-150ms), N2 (150-250ms), P2 (250-500ms)
    6. N2/P2 peak amplitudes, latencies, and N2-P2 peak-to-peak
    7. Vertex-channel Global Field Power
    8. Covariance matrix features (upper triangle)
    9. Downsampled post-stimulus waveform
    """
    data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_channels, n_times = data.shape
    times = epochs.times
    ch_names = epochs.info["ch_names"]

    feature_blocks = []

    # ── 1. Bandpower features (absolute + relative) ──
    n_fft = min(int(sfreq * 1.0), n_times)  # longer window for better freq resolution
    psds, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=FILTER_LOW, fmax=FILTER_HIGH,
        n_fft=n_fft, verbose=False,
    )
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = np.sum(psds, axis=2) * freq_res + 1e-10  # (n_epochs, n_channels)

    bp_blocks = []
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        bp = np.sum(psds[:, :, mask], axis=2) * freq_res
        log_bp = np.log10(bp + 1e-10)
        rel_bp = bp / total_power  # relative bandpower (normalised across subjects)
        bp_blocks.append(bp)
        feature_blocks.append(log_bp)       # absolute
        feature_blocks.append(rel_bp)       # relative

    # ── 2. Band-power ratios ──
    alpha_bp = bp_blocks[2]  # alpha
    beta_bp = bp_blocks[3] + 1e-10  # beta
    theta_bp = bp_blocks[1]  # theta
    feature_blocks.append(np.log10(alpha_bp / beta_bp + 1e-10))   # alpha/beta
    feature_blocks.append(np.log10(theta_bp / beta_bp + 1e-10))   # theta/beta
    feature_blocks.append(np.log10(theta_bp / (alpha_bp + 1e-10) + 1e-10))  # theta/alpha

    # ── 3. Pre-stimulus alpha power (pain anticipation) ──
    pre_stim_mask = (times >= -0.5) & (times < -0.1)
    if np.any(pre_stim_mask):
        pre_data = data[:, :, pre_stim_mask]
        pre_n_fft = min(int(sfreq * 0.4), pre_data.shape[2])
        if pre_n_fft >= 4:
            try:
                pre_psd, pre_freqs = psd_array_welch(
                    pre_data, sfreq=sfreq, fmin=8, fmax=13,
                    n_fft=pre_n_fft, verbose=False,
                )
                pre_alpha = np.log10(np.mean(pre_psd, axis=2) + 1e-10)
                feature_blocks.append(pre_alpha)
            except Exception:
                feature_blocks.append(np.zeros((n_epochs, n_channels)))
        else:
            feature_blocks.append(np.zeros((n_epochs, n_channels)))

    # ── 4. Temporal statistics (post-stimulus) ──
    t0_sample = np.argmin(np.abs(times - 0.0))
    post_data = data[:, :, t0_sample:]

    feature_blocks.append(np.mean(post_data, axis=2))
    feature_blocks.append(np.var(post_data, axis=2))
    feature_blocks.append(np.ptp(post_data, axis=2))
    feature_blocks.append(np.sqrt(np.mean(post_data ** 2, axis=2)))  # RMS

    # Skewness and kurtosis (vectorized)
    mu = np.mean(post_data, axis=2, keepdims=True)
    sigma = np.std(post_data, axis=2, keepdims=True) + 1e-10
    normed = (post_data - mu) / sigma
    feature_blocks.append(np.mean(normed ** 3, axis=2))   # skewness
    feature_blocks.append(np.mean(normed ** 4, axis=2))   # kurtosis

    # ── 5. ERP amplitude windows ──
    n1_mask = (times >= 0.05) & (times <= 0.15)   # N1 / A-delta
    n2_mask = (times >= 0.15) & (times <= 0.30)   # N2 (wider for variability)
    p2_mask = (times >= 0.25) & (times <= 0.50)   # P2 (wider window)
    late_mask = (times >= 0.50) & (times <= 0.90) # Late component

    for mask_name, mask in [("N1", n1_mask), ("N2", n2_mask), ("P2", p2_mask), ("late", late_mask)]:
        if np.any(mask):
            windowed = data[:, :, mask]
            feature_blocks.append(np.mean(windowed, axis=2))
            if mask_name == "N2":
                feature_blocks.append(np.min(windowed, axis=2))   # N2 = negative peak
            elif mask_name == "P2":
                feature_blocks.append(np.max(windowed, axis=2))   # P2 = positive peak
            elif mask_name == "late":
                feature_blocks.append(np.max(np.abs(windowed), axis=2))

    # ── 6. N2-P2 peak amplitude, latency, and peak-to-peak ──
    if np.any(n2_mask) and np.any(p2_mask):
        n2_data = data[:, :, n2_mask]
        p2_data = data[:, :, p2_mask]

        n2_min_val = np.min(n2_data, axis=2)
        p2_max_val = np.max(p2_data, axis=2)
        feature_blocks.append(p2_max_val - n2_min_val)  # N2-P2 amplitude

        # N2 latency (index of min within the N2 window)
        n2_times = times[n2_mask]
        n2_lat_idx = np.argmin(n2_data, axis=2)
        n2_latency = np.take(n2_times, n2_lat_idx)
        feature_blocks.append(n2_latency)

        # P2 latency (index of max within the P2 window)
        p2_times = times[p2_mask]
        p2_lat_idx = np.argmax(p2_data, axis=2)
        p2_latency = np.take(p2_times, p2_lat_idx)
        feature_blocks.append(p2_latency)

    # ── 7. Vertex-channel GFP ──
    vertex_idx = [i for i, ch in enumerate(ch_names) if ch in VERTEX_CHANNELS]
    if len(vertex_idx) >= 2:
        vertex_data = data[:, vertex_idx, :]
        # GFP = std across vertex channels at each time point
        gfp = np.std(vertex_data, axis=1)  # (n_epochs, n_times)
        # GFP features: mean, max, time of max (post-stimulus)
        gfp_post = gfp[:, t0_sample:]
        feature_blocks.append(np.mean(gfp_post, axis=1, keepdims=True))
        feature_blocks.append(np.max(gfp_post, axis=1, keepdims=True))
        gfp_peak_idx = np.argmax(gfp_post, axis=1)
        post_times = times[t0_sample:]
        gfp_peak_latency = np.take(post_times, gfp_peak_idx).reshape(-1, 1)
        feature_blocks.append(gfp_peak_latency)
        # Mean vertex amplitude in key windows
        for mask in [n2_mask, p2_mask]:
            if np.any(mask):
                feature_blocks.append(np.mean(vertex_data[:, :, mask], axis=(1, 2)).reshape(-1, 1))

    # ── 8. Morlet wavelet time-frequency features ──
    # Extract power in pain-relevant bands at key time windows using wavelets
    # This captures time-varying spectral content that Welch PSD misses
    wavelet_bands = [(4, 8), (8, 13), (13, 30), (30, 45)]  # theta, alpha, beta, gamma
    wavelet_windows = [
        ("early", 0.0, 0.3),    # early cortical response (N2)
        ("mid", 0.2, 0.6),      # N2-P2 complex
        ("late", 0.5, 1.0),     # late component
    ]
    for fmin, fmax in wavelet_bands:
        try:
            for win_name, t_start, t_end in wavelet_windows:
                win_mask = (times >= t_start) & (times <= t_end)
                if not np.any(win_mask):
                    continue
                win_data = data[:, :, win_mask]  # (n_epochs, n_ch, n_win)
                n_win_samples = win_data.shape[2]
                # Skip if window is too short for reliable filtering
                if n_win_samples < 50:
                    continue
                # Band-filtered power in time window (fast vectorized approach)
                # Use short filter to avoid filter_length > signal warnings
                flen = min(n_win_samples - 2, int(sfreq / fmin) * 3)
                if flen < 3:
                    continue
                flen = flen if flen % 2 == 1 else flen - 1  # must be odd
                win_bp = mne.filter.filter_data(
                    win_data.astype(np.float64), sfreq, fmin, fmax,
                    verbose=False, filter_length=flen,
                )
                win_power = np.log10(np.mean(win_bp ** 2, axis=2) + 1e-10)
                feature_blocks.append(win_power)
        except Exception:
            pass  # Skip if filtering fails for this band

    # ── 9. Vertex-focused ERP template matching ──
    # Average ERP amplitude at vertex channels in fine-grained time windows
    if len(vertex_idx) >= 2:
        vertex_post = data[:, vertex_idx, t0_sample:]  # (n_epochs, n_vertex, n_post)
        post_times_arr = times[t0_sample:]
        fine_windows = [
            (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
            (0.4, 0.6), (0.6, 0.8), (0.8, 1.0),
        ]
        for w_start, w_end in fine_windows:
            w_mask = (post_times_arr >= w_start) & (post_times_arr < w_end)
            if np.any(w_mask):
                w_mean = np.mean(vertex_post[:, :, w_mask], axis=2)  # (n_epochs, n_vertex)
                # Average across vertex channels for a robust scalar per window
                feature_blocks.append(np.mean(w_mean, axis=1, keepdims=True))
                # Also keep per-vertex info for top channels
                feature_blocks.append(w_mean)

    # ── 10. Hjorth parameters (activity, mobility, complexity) ──
    diff1 = np.diff(post_data, axis=2)
    diff2 = np.diff(diff1, axis=2)
    activity = np.var(post_data, axis=2)  # already have this as var, but needed for ratios
    var_diff1 = np.var(diff1, axis=2)
    mobility = np.sqrt(var_diff1 / (activity + 1e-10))
    var_diff2 = np.var(diff2, axis=2)
    complexity = np.sqrt(var_diff2 / (var_diff1 + 1e-10)) / (mobility + 1e-10)
    feature_blocks.append(mobility)
    feature_blocks.append(complexity)

    # ── 11. Spectral entropy ──
    psd_norm = psds / (np.sum(psds, axis=2, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=2)
    feature_blocks.append(spectral_entropy)

    # ── 12. Zero-crossing rate (post-stimulus) ──
    sign_changes = np.diff(np.sign(post_data), axis=2)
    zcr = np.sum(np.abs(sign_changes) > 0, axis=2) / max(post_data.shape[2] - 1, 1)
    feature_blocks.append(zcr)

    # ── 13. Lateralization index (C3 vs C4) ──
    c3_idx = [i for i, ch in enumerate(ch_names) if ch == "C3"]
    c4_idx = [i for i, ch in enumerate(ch_names) if ch == "C4"]
    if c3_idx and c4_idx:
        c3_power = np.mean(data[:, c3_idx[0], :] ** 2, axis=1, keepdims=True)
        c4_power = np.mean(data[:, c4_idx[0], :] ** 2, axis=1, keepdims=True)
        lateralization = (c4_power - c3_power) / (c4_power + c3_power + 1e-10)
        feature_blocks.append(lateralization)

    # ── 14. Phase Locking Value (PLV) connectivity features ──
    # PLV captures inter-channel phase synchronisation — a key pain biomarker.
    # We compute PLV in alpha (8-13 Hz) and beta (13-30 Hz) bands over the
    # post-stimulus window for key electrode pairs relevant to pain processing.
    key_pairs = [
        ("Cz", "FCz"), ("Cz", "Fz"), ("Cz", "C3"), ("Cz", "C4"),
        ("Cz", "Pz"), ("FCz", "Fz"), ("FCz", "C3"), ("C3", "C4"),
    ]
    valid_pairs = [(c1, c2) for c1, c2 in key_pairs
                   if c1 in ch_names and c2 in ch_names]
    if valid_pairs and n_times >= 100:
        # Cap filter length to fit within the epoch (must be odd and < n_times)
        plv_filter_len = int(sfreq * 0.8)  # 200 samples @ 250 Hz
        plv_filter_len = min(plv_filter_len, n_times - 2)
        if plv_filter_len % 2 == 0:
            plv_filter_len -= 1
        for fmin_plv, fmax_plv in [(8, 13), (13, 30)]:
            try:
                # Filter the FULL epoch (more samples → stable filter), then slice
                full_filt = mne.filter.filter_data(
                    data.astype(np.float64), sfreq, fmin_plv, fmax_plv,
                    filter_length=plv_filter_len, verbose=False,
                )
                plv_filt = full_filt[:, :, t0_sample:]  # post-stimulus only
                analytic = scipy_hilbert(plv_filt, axis=2)
                phase = np.angle(analytic)  # (n_epochs, n_channels, n_post_times)
                for c1_name, c2_name in valid_pairs:
                    ci = ch_names.index(c1_name)
                    cj = ch_names.index(c2_name)
                    plv = np.abs(
                        np.mean(np.exp(1j * (phase[:, ci, :] - phase[:, cj, :])), axis=1)
                    )
                    feature_blocks.append(plv.reshape(-1, 1))
            except Exception:
                pass

    features = np.hstack(feature_blocks)
    return features


# ── Subject Loader ───────────────────────────────────────────────────────────
def load_subject(
    subject_dir: str, debug: bool = False, apply_ica: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and process all runs for a single subject.
    Returns (features, labels, groups).
    """
    subject_id = Path(subject_dir).name
    t0 = time.time()
    logger.info(f"Processing {subject_id}...")

    vhdr_files = get_run_files(subject_dir)
    if not vhdr_files:
        logger.warning(f"No vhdr files found for {subject_id}")
        return np.array([]), np.array([]), np.array([])

    all_features = []
    all_labels = []

    for vhdr_path in vhdr_files:
        run_name = Path(vhdr_path).stem
        logger.debug(f"  Loading {run_name}")

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        except Exception as e:
            logger.warning(f"  Failed to load {run_name}: {e}")
            continue

        bad_channels = load_bad_channels(vhdr_path)

        try:
            raw_processed = preprocess_raw(raw, bad_channels, apply_ica=apply_ica)
        except Exception as e:
            logger.warning(f"  Failed to preprocess {run_name}: {e}")
            continue

        events_tsv = vhdr_path.replace("_eeg.vhdr", "_events.tsv")
        if not os.path.exists(events_tsv):
            logger.warning(f"  Events file not found: {events_tsv}")
            continue

        try:
            epochs, labels = create_epochs(raw_processed, events_tsv, debug=debug)
        except Exception as e:
            logger.warning(f"  Failed to create epochs for {run_name}: {e}")
            continue

        if len(epochs) == 0:
            logger.warning(f"  No valid epochs for {run_name}")
            continue

        try:
            features = extract_features(epochs)
        except Exception as e:
            logger.warning(f"  Failed to extract features for {run_name}: {e}")
            continue

        all_features.append(features)
        all_labels.append(labels)
        del raw, raw_processed, epochs

    if not all_features:
        return np.array([]), np.array([]), np.array([])

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    groups = np.full(len(labels), subject_id)

    # ── Per-subject z-normalization ──
    # Critical for cross-subject generalisation: each subject has different
    # EEG amplitudes, impedances, and baseline neural activity.  Normalising
    # features to zero mean / unit variance *within* each subject removes
    # these nuisance differences so the classifier learns pain-vs-baseline
    # patterns instead of subject identity.
    if len(features) > 1:
        subj_mean = np.nanmean(features, axis=0, keepdims=True)
        subj_std = np.nanstd(features, axis=0, keepdims=True) + 1e-10
        features = (features - subj_mean) / subj_std

    elapsed = time.time() - t0
    logger.info(
        f"  {subject_id}: {len(labels)} epochs "
        f"(pain={np.sum(labels == 1)}, baseline={np.sum(labels == 0)}) "
        f"in {elapsed:.1f}s"
    )
    return features, labels, groups


# ── Model Training ───────────────────────────────────────────────────────────
def get_classifiers(n_features: int, random_state: int = 42) -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines to compare."""
    n_pca = min(80, n_features)  # PCA for dimensionality reduction
    n_select = min(300, n_features)  # feature selection before PCA
    n_select_tree = min(150, n_features)  # fewer features for tree models (less overfit)

    clfs = {}

    # LDA with shrinkage + feature selection
    clfs["LDA"] = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=n_select)),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

    # Logistic Regression variants
    for C in [0.01, 0.1, 1.0]:
        clfs[f"LogReg_C{C}"] = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=n_select)),
            ("pca", PCA(n_components=n_pca, random_state=random_state)),
            ("clf", LogisticRegression(
                max_iter=3000, solver="lbfgs",
                class_weight="balanced", C=C,
                random_state=random_state,
            )),
        ])

    # SVM with RBF kernel — strong baseline for EEG BCI
    clfs["SVM_RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=n_select)),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("clf", SVC(
            kernel="rbf", C=5.0, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=random_state,
        )),
    ])

    # Random Forest — robust to noisy EEG features
    clfs["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=n_select_tree)),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),
    ])

    # Extra Trees — fast, low-variance ensemble
    clfs["ExtraTrees"] = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=n_select_tree)),
        ("clf", ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),
    ])

    # LightGBM — moderately regularised for small EEG datasets
    try:
        from lightgbm import LGBMClassifier
        clfs["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=n_select_tree)),
            ("clf", LGBMClassifier(
                n_estimators=600, max_depth=5, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.6,
                min_child_samples=20, reg_alpha=0.5, reg_lambda=1.5,
                class_weight="balanced", random_state=random_state,
                n_jobs=-1, verbose=-1,
            )),
        ])
    except ImportError:
        logger.info("LightGBM not installed - skipping")

    # XGBoost — moderately regularised for small EEG datasets
    try:
        from xgboost import XGBClassifier
        clfs["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=n_select_tree)),
            ("clf", XGBClassifier(
                n_estimators=600, max_depth=5, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.6,
                min_child_weight=5, reg_alpha=0.5, reg_lambda=1.5,
                scale_pos_weight=1.0,
                eval_metric="logloss",
                random_state=random_state, n_jobs=-1,
            )),
        ])
    except ImportError:
        logger.info("XGBoost not installed - skipping")

    # Soft Voting Ensemble — diverse models: linear, kernel, tree, boosting
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        vote_estimators = [
            ("lda", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select)),
                ("pca", PCA(n_components=n_pca, random_state=random_state)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ])),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select)),
                ("pca", PCA(n_components=n_pca, random_state=random_state)),
                ("clf", SVC(
                    kernel="rbf", C=5.0, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=random_state,
                )),
            ])),
            ("lgbm", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select_tree)),
                ("clf", LGBMClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.6,
                    class_weight="balanced", reg_alpha=0.5, reg_lambda=1.5,
                    random_state=random_state, n_jobs=-1, verbose=-1,
                )),
            ])),
            ("rf", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select_tree)),
                ("clf", RandomForestClassifier(
                    n_estimators=400, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=random_state, n_jobs=-1,
                )),
            ])),
        ]

        clfs["SoftVoting"] = VotingClassifier(
            estimators=vote_estimators,
            voting="soft",
            n_jobs=1,
        )
    except ImportError:
        logger.info("Voting requires LightGBM - skipping")

    # Stacking Ensemble (meta-learner) — LDA + SVM + LightGBM → LogReg
    try:
        from lightgbm import LGBMClassifier

        stack_estimators = [
            ("lda", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select)),
                ("pca", PCA(n_components=n_pca, random_state=random_state)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ])),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select)),
                ("pca", PCA(n_components=n_pca, random_state=random_state)),
                ("clf", SVC(
                    kernel="rbf", C=5.0, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=random_state,
                )),
            ])),
            ("lgbm", Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select_tree)),
                ("clf", LGBMClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.02,
                    subsample=0.8, colsample_bytree=0.6,
                    class_weight="balanced", reg_alpha=0.5, reg_lambda=1.5,
                    random_state=random_state, n_jobs=-1, verbose=-1,
                )),
            ])),
        ]

        clfs["StackingEnsemble"] = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(
                max_iter=2000, solver="lbfgs", C=1.0,
                class_weight="balanced", random_state=random_state,
            ),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            n_jobs=1,
        )
    except ImportError:
        logger.info("Stacking requires LightGBM - skipping")

    return clfs


def train_model(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    test_size: float = 0.2, random_state: int = 42,
) -> dict:
    """
    Train and evaluate multiple classifiers.

    Split strategy:
    - >= 3 subjects -> Leave-One-Subject-Out CV (gold standard)
    - 2 subjects    -> GroupShuffleSplit
    - 1 subject     -> random split (sanity check only)
    """
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)

    # Remove NaN/Inf rows
    valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    if not np.all(valid_mask):
        n_bad = np.sum(~valid_mask)
        logger.info(f"Dropping {n_bad} rows with NaN/Inf features ({n_bad/len(valid_mask)*100:.1f}%)")
        X, y, groups = X[valid_mask], y[valid_mask], groups[valid_mask]

    # Apply SMOTE to balance classes (important for this imbalanced dataset)
    if HAS_SMOTE:
        n_minority = np.sum(y == 0)
        n_majority = np.sum(y == 1)
        if n_minority > 5 and n_majority > n_minority * 1.5:
            logger.info(f"Applying SMOTE: {n_minority} minority -> {n_majority} (target 1:1)")
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=min(5, n_minority - 1))
                X, y = smote.fit_resample(X, y)
                # Expand groups array for synthetic samples
                groups = np.concatenate([groups, np.full(len(y) - len(groups), "synthetic")])
                logger.info(f"After SMOTE: {np.sum(y==0)} baseline, {np.sum(y==1)} pain")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")

    clfs = get_classifiers(n_features=X.shape[1], random_state=random_state)
    best_name, best_acc, best_result = None, -1.0, None
    all_results = {}

    if n_subjects >= 3:
        logger.info(f"Leave-One-Subject-Out CV across {n_subjects} subjects")
        logo = LeaveOneGroupOut()

        for name, pipe in clfs.items():
            t0 = time.time()
            try:
                # Use n_jobs=-1 for non-stacking models; stacking has internal CV
                # that can deadlock with nested parallelism
                n_jobs_cv = 1 if "Stacking" in name else -1
                y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=logo, n_jobs=n_jobs_cv)
            except Exception as e:
                logger.warning(f"  {name:25s}  FAILED: {e}")
                continue
            elapsed = time.time() - t0
            acc = accuracy_score(y, y_pred)
            bal_acc = balanced_accuracy_score(y, y_pred)
            try:
                auc = roc_auc_score(y, y_pred)
            except ValueError:
                auc = float("nan")
            logger.info(f"  {name:25s}  acc={acc:.4f}  bal_acc={bal_acc:.4f}  auc={auc:.4f}  ({elapsed:.1f}s)")

            all_results[name] = {"accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc, "time": elapsed}

            # Select by BALANCED accuracy (prevents majority-class bias)
            if bal_acc > best_acc:
                best_acc = bal_acc
                best_name = name
                best_result = {
                    "model_name": name,
                    "pipeline": pipe,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "auc": auc,
                    "classification_report": classification_report(
                        y, y_pred, target_names=["baseline", "pain"],
                    ),
                    "confusion_matrix": confusion_matrix(y, y_pred),
                    "y_test": y,
                    "y_pred": y_pred,
                    "all_results": all_results,
                }

    elif n_subjects == 2:
        logger.info("GroupShuffleSplit (2 subjects)")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        logger.info(f"Train subjects: {sorted(set(groups[train_idx]))}")
        logger.info(f"Test subjects:  {sorted(set(groups[test_idx]))}")

        for name, pipe in clfs.items():
            t0 = time.time()
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            except Exception as e:
                logger.warning(f"  {name:25s}  FAILED: {e}")
                continue
            elapsed = time.time() - t0
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"  {name:25s}  acc={acc:.4f}  ({elapsed:.1f}s)")

            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_result = {
                    "model_name": name,
                    "pipeline": pipe,
                    "accuracy": acc,
                    "classification_report": classification_report(
                        y_test, y_pred, target_names=["baseline", "pain"],
                    ),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "y_test": y_test,
                    "y_pred": y_pred,
                }
    else:
        logger.warning("Only 1 subject - random split (NOT VALID for real evaluation)")
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=y,
            random_state=random_state,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, pipe in clfs.items():
            t0 = time.time()
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            except Exception as e:
                logger.warning(f"  {name:25s}  FAILED: {e}")
                continue
            elapsed = time.time() - t0
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"  {name:25s}  acc={acc:.4f}  ({elapsed:.1f}s)")

            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_result = {
                    "model_name": name,
                    "pipeline": pipe,
                    "accuracy": acc,
                    "classification_report": classification_report(
                        y_test, y_pred, target_names=["baseline", "pain"],
                    ),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "y_test": y_test,
                    "y_pred": y_pred,
                }

    return best_result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EEG Pain Classification Pipeline v3 for ds005307"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ds005307 dataset directory")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test proportion (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Max subjects to process (for testing)")
    parser.add_argument("--no_ica", action="store_true",
                        help="Disable ICA artifact removal (faster)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export best model to this file path (e.g. pain_model.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    t_start = time.time()

    logger.info("=" * 60)
    logger.info("EEG Pain Classification Pipeline v3")
    logger.info("=" * 60)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    subject_dirs = get_subject_dirs(args.data_dir)
    if not subject_dirs:
        logger.error("No subject directories found")
        sys.exit(1)

    logger.info(f"Found {len(subject_dirs)} subjects")

    if args.max_subjects:
        subject_dirs = subject_dirs[: args.max_subjects]
        logger.info(f"Processing first {args.max_subjects} subjects only")

    logger.info("-" * 60)
    logger.info("Loading and preprocessing EEG data...")
    logger.info("-" * 60)

    all_features, all_labels, all_groups = [], [], []
    apply_ica = not args.no_ica
    n_features_ref = None

    for i, subject_dir in enumerate(subject_dirs):
        debug_this = args.verbose and (i == 0)
        features, labels, groups = load_subject(
            subject_dir, debug=debug_this, apply_ica=apply_ica,
        )
        if len(features) > 0:
            # Ensure consistent feature dimensions across subjects
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
        logger.error("No valid data extracted from any subject")
        sys.exit(1)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    # ── CSP features removed from global scope to avoid data leakage ──
    # Fitting CSP on the entire dataset before cross-validation leaks test-fold
    # information into the spatial filters, inflating accuracy. CSP features
    # should be computed per-fold inside the CV loop for a rigorous evaluation.
    # For now we proceed without CSP; the hand-crafted features are sufficient.
    logger.info("CSP features skipped (global fit causes data leakage in CV)")

    logger.info("-" * 60)
    logger.info("Dataset Summary")
    logger.info("-" * 60)
    logger.info(f"Total epochs:     {len(y)}")
    logger.info(f"Pain epochs:      {np.sum(y == 1)}")
    logger.info(f"Baseline epochs:  {np.sum(y == 0)}")
    logger.info(f"Feature dims:     {X.shape}")
    logger.info(f"Unique subjects:  {len(np.unique(groups))}")

    logger.info("-" * 60)
    logger.info("Training & evaluating classifiers...")
    logger.info("-" * 60)

    results = train_model(
        X, y, groups,
        test_size=args.test_size, random_state=args.random_state,
    )

    total_time = time.time() - t_start

    logger.info("=" * 60)
    logger.info("BEST MODEL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model:     {results['model_name']}")
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    if "auc" in results:
        logger.info(f"AUC:       {results['auc']:.4f}")
    logger.info(f"\nClassification Report:\n{results['classification_report']}")
    logger.info(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    logger.info(f"\nTotal pipeline time: {total_time:.1f}s")

    # Export model
    if args.export:
        logger.info(f"\nExporting best model to {args.export}...")
        # Refit on all data
        pipe = results["pipeline"]
        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        X_clean, y_clean = X[valid_mask], y[valid_mask]
        pipe.fit(X_clean, y_clean)
        export_data = {
            "pipeline": pipe,
            "model_name": results["model_name"],
            "accuracy": results["accuracy"],
            "scalp_channels": SCALP_CHANNELS,
            "resample_freq": RESAMPLE_FREQ,
            "filter_low": FILTER_LOW,
            "filter_high": FILTER_HIGH,
            "tmin": TMIN,
            "tmax": TMAX,
            "n_features": X_clean.shape[1],
        }
        joblib.dump(export_data, args.export)
        logger.info(f"Model exported to {args.export}")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()