#!/usr/bin/env python3
"""
Live EEG Pain Inference
Loads a trained model from train_baseline.py and classifies
incoming EEG data in real-time from a BrainVision LSL stream
or from a file (for testing).

Usage:
    # From file (test mode):
    python live_inference.py --model pain_model.pkl --file test_data.vhdr

    # From LSL stream (live headset):
    python live_inference.py --model pain_model.pkl --lsl

Requirements:
    pip install joblib mne numpy
    pip install pylsl  # for live LSL streaming
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import mne
import numpy as np
from mne.time_frequency import psd_array_welch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")

# ── Constants (must match training) ──────────────────────────────────────────
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

VERTEX_CHANNELS = ["Cz", "FCz", "Fz", "C3", "C4", "FC1", "FC2", "CP1", "CP2"]

SCALP_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FCz", "Fz", "Cz", "Pz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
    "FT9", "FT10", "TP9", "TP10",
]


def load_model(model_path: str) -> dict:
    """Load exported model and metadata."""
    data = joblib.load(model_path)
    logger.info(f"Loaded model: {data['model_name']} (acc={data['accuracy']:.4f})")
    logger.info(f"  Channels: {len(data['scalp_channels'])}, Features: {data['n_features']}")
    return data


def preprocess_epoch(epoch_data: np.ndarray, sfreq: float, model_meta: dict) -> np.ndarray:
    """
    Preprocess a single epoch array for inference.
    epoch_data shape: (n_channels, n_times)
    Returns: feature vector (1, n_features)
    """
    filter_low = model_meta.get("filter_low", 0.5)
    filter_high = model_meta.get("filter_high", 45.0)
    resample_freq = model_meta.get("resample_freq", 250.0)
    tmin = model_meta.get("tmin", -0.5)
    tmax = model_meta.get("tmax", 1.0)

    n_channels, n_times = epoch_data.shape
    epoch_3d = epoch_data[np.newaxis, :, :]  # (1, n_ch, n_times)

    # Construct time array
    total_dur = tmax - tmin
    times = np.linspace(tmin, tmax, n_times)

    feature_blocks = []

    # ── Bandpower ──
    n_fft = min(int(sfreq * 1.0), n_times)
    psds, freqs = psd_array_welch(
        epoch_3d, sfreq=sfreq, fmin=filter_low, fmax=filter_high,
        n_fft=n_fft, verbose=False,
    )
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = np.sum(psds, axis=2) * freq_res + 1e-10

    bp_blocks = []
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        bp = np.sum(psds[:, :, mask], axis=2) * freq_res
        log_bp = np.log10(bp + 1e-10)
        rel_bp = bp / total_power
        bp_blocks.append(bp)
        feature_blocks.append(log_bp)
        feature_blocks.append(rel_bp)

    # Band ratios
    alpha_bp = bp_blocks[2]
    beta_bp = bp_blocks[3] + 1e-10
    theta_bp = bp_blocks[1]
    feature_blocks.append(np.log10(alpha_bp / beta_bp + 1e-10))
    feature_blocks.append(np.log10(theta_bp / beta_bp + 1e-10))
    feature_blocks.append(np.log10(theta_bp / (alpha_bp + 1e-10) + 1e-10))

    # Pre-stimulus alpha
    pre_stim_mask = (times >= -0.5) & (times < -0.1)
    if np.any(pre_stim_mask):
        pre_data = epoch_3d[:, :, pre_stim_mask]
        pre_n_fft = min(int(sfreq * 0.4), pre_data.shape[2])
        if pre_n_fft >= 4:
            try:
                pre_psd, _ = psd_array_welch(
                    pre_data, sfreq=sfreq, fmin=8, fmax=13,
                    n_fft=pre_n_fft, verbose=False,
                )
                feature_blocks.append(np.log10(np.mean(pre_psd, axis=2) + 1e-10))
            except Exception:
                feature_blocks.append(np.zeros((1, n_channels)))
        else:
            feature_blocks.append(np.zeros((1, n_channels)))

    # Temporal stats (post-stimulus)
    t0_sample = np.argmin(np.abs(times - 0.0))
    post_data = epoch_3d[:, :, t0_sample:]

    feature_blocks.append(np.mean(post_data, axis=2))
    feature_blocks.append(np.var(post_data, axis=2))
    feature_blocks.append(np.ptp(post_data, axis=2))
    feature_blocks.append(np.sqrt(np.mean(post_data ** 2, axis=2)))

    mu = np.mean(post_data, axis=2, keepdims=True)
    sigma = np.std(post_data, axis=2, keepdims=True) + 1e-10
    normed = (post_data - mu) / sigma
    feature_blocks.append(np.mean(normed ** 3, axis=2))
    feature_blocks.append(np.mean(normed ** 4, axis=2))

    # ERP windows
    n1_mask = (times >= 0.05) & (times <= 0.15)
    n2_mask = (times >= 0.15) & (times <= 0.30)
    p2_mask = (times >= 0.25) & (times <= 0.50)
    late_mask = (times >= 0.50) & (times <= 0.90)

    data = epoch_3d
    for mask_name, mask in [("N1", n1_mask), ("N2", n2_mask), ("P2", p2_mask), ("late", late_mask)]:
        if np.any(mask):
            windowed = data[:, :, mask]
            feature_blocks.append(np.mean(windowed, axis=2))
            if mask_name == "N2":
                feature_blocks.append(np.min(windowed, axis=2))
            elif mask_name == "P2":
                feature_blocks.append(np.max(windowed, axis=2))
            elif mask_name == "late":
                feature_blocks.append(np.max(np.abs(windowed), axis=2))

    # N2-P2 features
    if np.any(n2_mask) and np.any(p2_mask):
        n2_data = data[:, :, n2_mask]
        p2_data = data[:, :, p2_mask]
        n2_min_val = np.min(n2_data, axis=2)
        p2_max_val = np.max(p2_data, axis=2)
        feature_blocks.append(p2_max_val - n2_min_val)

        n2_times = times[n2_mask]
        n2_lat_idx = np.argmin(n2_data, axis=2)
        feature_blocks.append(np.take(n2_times, n2_lat_idx))

        p2_times = times[p2_mask]
        p2_lat_idx = np.argmax(p2_data, axis=2)
        feature_blocks.append(np.take(p2_times, p2_lat_idx))

    # GFP (placeholder vertex indices - will match whatever channels are available)
    # Skip if we don't have enough channels

    # Covariance
    if n_channels <= 32:
        n_upper = n_channels * (n_channels - 1) // 2
        cov = np.cov(post_data[0])
        trace = np.trace(cov) + 1e-10
        cov_norm = cov / trace
        cov_feats = cov_norm[np.triu_indices(n_channels, k=1)].reshape(1, -1)
        feature_blocks.append(cov_feats)

    # Downsampled waveform
    n_post_samples = post_data.shape[2]
    n_downsample = min(15, n_post_samples)
    if n_downsample > 0:
        indices = np.linspace(0, n_post_samples - 1, n_downsample, dtype=int)
        waveform = post_data[:, :, indices].reshape(1, -1)
        feature_blocks.append(waveform)

    features = np.hstack(feature_blocks)
    return features


def classify_epoch(features: np.ndarray, model_data: dict) -> tuple[int, float]:
    """
    Classify a single epoch.
    Returns (prediction, confidence).
    """
    pipeline = model_data["pipeline"]
    pred = pipeline.predict(features)[0]

    # Try to get probability
    try:
        proba = pipeline.predict_proba(features)[0]
        confidence = max(proba)
    except Exception:
        confidence = float("nan")

    return int(pred), confidence


def run_file_mode(model_data: dict, file_path: str):
    """Test inference on a BrainVision file."""
    logger.info(f"Loading test file: {file_path}")

    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)

    # Preprocess
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    scalp = [ch for ch in SCALP_CHANNELS if ch in raw.ch_names]
    if scalp:
        raw.pick(scalp)

    sfreq_orig = raw.info["sfreq"]
    resample_freq = model_data["resample_freq"]
    tmin = model_data["tmin"]
    tmax = model_data["tmax"]

    raw.filter(
        l_freq=model_data["filter_low"], h_freq=model_data["filter_high"],
        fir_design="firwin", verbose=False,
    )
    raw.notch_filter(freqs=60.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(resample_freq, verbose=False)

    # Load events
    events_tsv = file_path.replace("_eeg.vhdr", "_events.tsv")
    if not Path(events_tsv).exists():
        logger.error(f"Events file not found: {events_tsv}")
        return

    events_df = pd.read_csv(events_tsv, sep="\t")
    import pandas as pd

    pain_mask = events_df["trial_type"].str.lower().str.contains("pain", na=False)
    pain_df = events_df[pain_mask]

    sfreq = raw.info["sfreq"]
    n_pain, n_correct = 0, 0

    for _, row in pain_df.iterrows():
        onset_sample = int(round(row["onset"] * sfreq))
        start = onset_sample + int(tmin * sfreq)
        end = onset_sample + int(tmax * sfreq)

        if start < 0 or end > raw.n_times:
            continue

        epoch_data = raw.get_data(start=start, stop=end)
        features = preprocess_epoch(epoch_data, sfreq, model_data)
        pred, conf = classify_epoch(features, model_data)

        label = "PAIN" if pred == 1 else "baseline"
        logger.info(f"  Trial {n_pain + 1}: {label} (conf={conf:.3f})")

        n_pain += 1
        if pred == 1:
            n_correct += 1

    logger.info(f"\nPain detection: {n_correct}/{n_pain} = {n_correct / max(n_pain, 1) * 100:.1f}%")


def run_lsl_mode(model_data: dict):
    """Live inference from LSL stream."""
    try:
        from pylsl import StreamInlet, resolve_stream
    except ImportError:
        logger.error("pylsl not installed. Run: pip install pylsl")
        sys.exit(1)

    logger.info("Looking for EEG LSL stream...")
    streams = resolve_stream("type", "EEG")
    if not streams:
        logger.error("No EEG stream found")
        sys.exit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    sfreq = info.nominal_srate()
    n_channels = info.channel_count()

    resample_freq = model_data["resample_freq"]
    tmin = model_data["tmin"]
    tmax = model_data["tmax"]
    epoch_samples = int((tmax - tmin) * resample_freq)

    logger.info(f"Connected: {n_channels} channels @ {sfreq} Hz")
    logger.info(f"Epoch window: {tmin} to {tmax}s ({epoch_samples} samples @ {resample_freq}Hz)")
    logger.info("Waiting for trigger events (marker stream)...")

    # Circular buffer for continuous data
    buffer_seconds = 5.0
    buffer_size = int(buffer_seconds * sfreq)
    buffer = np.zeros((n_channels, buffer_size))
    buf_idx = 0

    # Look for marker stream
    marker_streams = resolve_stream("type", "Markers")
    marker_inlet = StreamInlet(marker_streams[0]) if marker_streams else None

    logger.info("Streaming... Press Ctrl+C to stop")
    try:
        while True:
            # Pull EEG samples
            sample, timestamp = inlet.pull_sample(timeout=0.01)
            if sample is not None:
                buffer[:, buf_idx % buffer_size] = sample[:n_channels]
                buf_idx += 1

            # Check for trigger
            if marker_inlet:
                marker, ts = marker_inlet.pull_sample(timeout=0.0)
                if marker is not None:
                    # Extract epoch around trigger
                    trigger_idx = buf_idx
                    pre_samples = int(abs(tmin) * sfreq)
                    post_samples = int(tmax * sfreq)
                    total = pre_samples + post_samples

                    if trigger_idx >= pre_samples:
                        start = (trigger_idx - pre_samples) % buffer_size
                        indices = [(start + i) % buffer_size for i in range(total)]
                        epoch = buffer[:, indices]

                        # Resample if needed
                        if sfreq != resample_freq:
                            ratio = resample_freq / sfreq
                            new_len = int(epoch.shape[1] * ratio)
                            from scipy.signal import resample as sci_resample
                            epoch = sci_resample(epoch, new_len, axis=1)

                        features = preprocess_epoch(epoch, resample_freq, model_data)
                        pred, conf = classify_epoch(features, model_data)

                        label = "** PAIN **" if pred == 1 else "baseline"
                        logger.info(f"[{time.strftime('%H:%M:%S')}] {label} (conf={conf:.3f})")

    except KeyboardInterrupt:
        logger.info("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Live EEG Pain Inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to exported model (.pkl)")
    parser.add_argument("--file", type=str, default=None,
                        help="BrainVision .vhdr file for offline testing")
    parser.add_argument("--lsl", action="store_true",
                        help="Use LSL stream for live inference")
    args = parser.parse_args()

    model_data = load_model(args.model)

    if args.file:
        run_file_mode(model_data, args.file)
    elif args.lsl:
        run_lsl_mode(model_data)
    else:
        logger.error("Specify --file or --lsl")
        sys.exit(1)


if __name__ == "__main__":
    main()
