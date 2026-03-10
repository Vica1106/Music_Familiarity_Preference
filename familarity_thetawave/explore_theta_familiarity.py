#!/usr/bin/env python3
"""
Exploratory analysis: Familiarity & Theta (methods adapted from enjoyment/alpha pipeline).

- Event-based epoching (music start from events.tsv), preprocess (1–50 Hz, average ref),
  Morlet TFR, and band power extraction — same style as the enjoyment/alpha code.
- Focus: familiarity (Low/High) and theta (4–8 Hz); alpha (8–13 Hz) for comparison.
- Outputs (in results/):
  1. explore_spectrogram_theta.png       — Time–frequency spectrogram (one channel).
  2. explore_all_bands_over_time.png    — All bands (delta, theta, alpha, beta, gamma) over time, same style as friend’s “Frequency Band Power Over Time”.
  3. explore_theta_alpha_over_time.png  — Theta & alpha only (frontal).
  4. explore_theta_by_region_over_time.png — Theta power over time by frontal/central/parietal.
  5. explore_theta_vs_familiarity_group.png — Boxplot theta vs Low/High (if --max-runs > 0).
  6. explore_topomap_familiarity_low_vs_high.png — Topomaps comparing theta: Low vs High familiarity (3 example trials per group, like Graph 7).

Usage:
  python explore_theta_familiarity.py
  python explore_theta_familiarity.py --sub 1 --ses 2
  python explore_theta_familiarity.py --max-runs 10   # also plot theta vs familiarity group
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mne
    from mne.time_frequency import tfr_morlet
    from mne.channels import make_dig_montage
    try:
        from mne import EvokedArray
    except ImportError:
        from mne.evoked import EvokedArray
    mne.set_log_level("WARNING")
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

# Paths (same as main analysis)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
BEHAV_PATH = DATA_ROOT / "stimuli" / "Behavioural_data"
EEG_TEMPLATE = "sub-{sub:03d}/ses-{ses:02d}/eeg/sub-{sub:03d}_ses-{ses:02d}_task-MusicListening_run-{run:d}_eeg.set"
EVENTS_TEMPLATE = "sub-{sub:03d}/ses-{ses:02d}/eeg/sub-{sub:03d}_ses-{ses:02d}_task-MusicListening_run-{run:d}_events.tsv"
ELECTRODES_TEMPLATE = "sub-{sub:03d}/ses-{ses:02d}/eeg/sub-{sub:03d}_ses-{ses:02d}_task-MusicListening_run-{run:d}_electrodes.tsv"
OUT_DIR = SCRIPT_DIR / "results"

# Frequency bands (theta focus; alpha for comparison)
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def load_behavioural():
    """Load behavioural data; keep Familiarity in {1,2,5}; add Group Low/High."""
    rows = []
    with open(BEHAV_PATH) as f:
        f.readline()
        for line in f:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 4:
                continue
            subject, song_id = int(parts[0]), int(parts[1])
            familiarity = int(parts[3])
            if familiarity not in (1, 2, 5):
                continue
            group = "Low" if familiarity in (1, 2) else "High"
            rows.append({
                "Subject": subject,
                "Song_ID": song_id,
                "Familiarity": familiarity,
                "Familiarity_Group": group,
            })
    return pd.DataFrame(rows)


def load_eeg_with_events(subject_id, song_id):
    """
    Load EEG and events for one run. Find music-start event (stim/clyp/stm+/fxcl).
    Returns raw, events_array (n_events x 3), event_id dict, or (None, None, None).
    """
    if not HAS_MNE:
        return None, None, None
    sub, ses = int(subject_id), int(song_id)
    eeg_path = DATA_ROOT / EEG_TEMPLATE.format(sub=sub, ses=ses, run=ses)
    events_path = DATA_ROOT / EVENTS_TEMPLATE.format(sub=sub, ses=ses, run=ses)
    if not eeg_path.exists() or not events_path.exists():
        return None, None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        events_df = pd.read_csv(events_path, sep="\t")
        duration = raw.times[-1]
        sfreq = raw.info["sfreq"]
        if "onset" in events_df.columns:
            in_range = events_df[(events_df["onset"] >= 0) & (events_df["onset"] < duration)]
        else:
            in_range = events_df[(events_df["sample"] / sfreq >= 0) & (events_df["sample"] / sfreq < duration)]
        if len(in_range) == 0:
            return raw, np.array([]), {}
        for event_type in ["stim", "clyp", "stm+", "fxcl"]:
            candidates = in_range[in_range["value"] == event_type]
            if len(candidates) > 0:
                first = candidates.iloc[0]
                if "onset" in first:
                    t = first["onset"]
                    sample = int(t * sfreq)
                else:
                    sample = int(first["sample"])
                    t = sample / sfreq
                if sample >= len(raw.times):
                    continue
                events_array = np.array([[sample, 0, 1]])
                event_id = {"music_start": 1}
                return raw, events_array, event_id
        first = in_range.iloc[0]
        t = first["onset"] if "onset" in first else first["sample"] / sfreq
        sample = int(t * sfreq) if "onset" in first else int(first["sample"])
        if sample >= len(raw.times):
            return raw, np.array([]), {}
        events_array = np.array([[sample, 0, 1]])
        event_id = {"music_start": 1}
        return raw, events_array, event_id
    except Exception:
        return None, None, None


def preprocess_raw(raw, l_freq=1.0, h_freq=50.0):
    """Filter and average reference."""
    if not HAS_MNE or raw is None:
        return raw
    raw = raw.copy()
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def create_music_epochs(raw, events, event_id, tmin=0.0, tmax=120.0):
    """Create one long epoch from music start. Falls back to fixed-length if events invalid."""
    if raw is None:
        return None
    duration = raw.times[-1]
    sfreq = raw.info["sfreq"]
    if events is None or len(events) == 0:
        events = mne.make_fixed_length_events(raw, id=1, duration=2.0, start=1.0, stop=duration - 1.0)
        event_id = {"music": 1}
    else:
        t0 = events[0, 0] / sfreq
        if t0 + tmax > duration:
            tmax = duration - t0 - 0.1
    try:
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=(tmin, 0) if tmin < 0 else None,
            preload=True, verbose=False,
        )
        return epochs if len(epochs) > 0 else None
    except Exception:
        events = mne.make_fixed_length_events(raw, id=1, duration=2.0, start=1.0, stop=duration - 1.0)
        event_id = {"music": 1}
        return mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False)


def make_montage_from_electrodes(electrodes_tsv):
    """Build MNE montage from BIDS electrodes.tsv (x,y,z in cm; E1–E128 only)."""
    ch_pos = {}
    with open(electrodes_tsv) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            if name == "E129":
                continue
            ch_pos[name] = np.array([x, y, z]) / 100.0
    return make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def get_electrode_regions(electrodes_tsv):
    """Channel name -> region (frontal/central/parietal) from z in electrodes.tsv."""
    ch_to_region = {}
    with open(electrodes_tsv) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name, z = parts[0], float(parts[3])
            if name == "E129":
                continue
            ch_to_region[name] = "frontal" if z >= 5 else ("central" if z >= -2 else "parietal")
    return ch_to_region


def compute_tfr(epochs, freqs=None, n_cycles=5, decim=4):
    """Morlet time-frequency; average over epochs."""
    if freqs is None:
        freqs = np.arange(2, 51, 2)
    n_cycles = np.clip(freqs / 2.0, 2, 10)
    return tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True, decim=decim, verbose=False,
    )


def extract_band_power(tfr, band_name, channels=None):
    """Average power in band: (n_channels, n_times) or subset of channels."""
    if band_name not in FREQ_BANDS:
        raise ValueError(f"Unknown band: {band_name}. Choose from {list(FREQ_BANDS.keys())}")
    fmin, fmax = FREQ_BANDS[band_name]
    freq_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    data = tfr.data[:, freq_mask, :]  # ch x freq x time
    if channels is not None:
        inds = [tfr.ch_names.index(c) for c in channels if c in tfr.ch_names]
        data = data[inds, :, :]
    return np.mean(data, axis=1)  # ch x time (or 1d if single ch)


def _theta_channel_power_for_topomap(subject_id, song_id, log_fn=None):
    """
    Load one run, set montage, preprocess, epoch, TFR; return (info, theta_per_ch) for topomap.
    theta_per_ch is mean theta power over time, one value per channel. Returns (None, None) on failure.
    """
    raw, events, event_id = load_eeg_with_events(subject_id, song_id)
    if raw is None:
        return None, None
    electrodes_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=subject_id, ses=song_id, run=song_id)
    if not electrodes_tsv.exists():
        return None, None
    try:
        montage = make_montage_from_electrodes(electrodes_tsv)
        ch_pick = [c for c in raw.ch_names if c in montage.ch_names]
        if len(ch_pick) < 10:
            return None, None
        raw.pick(ch_pick)
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        raw = preprocess_raw(raw)
        epochs = create_music_epochs(raw, events, event_id, tmin=0.0, tmax=min(120.0, raw.times[-1] - 1))
        if epochs is None or len(epochs) == 0:
            return None, None
        power_tfr = compute_tfr(epochs, freqs=np.arange(2, 51, 2), decim=4)
        theta_2d = extract_band_power(power_tfr, "theta")  # (n_ch, n_times)
        theta_per_ch = np.mean(theta_2d, axis=1)
        return power_tfr.info, theta_per_ch
    except Exception as e:
        if log_fn:
            log_fn(f"  Topomap example Subj{subject_id} Song{song_id}: {e}")
        return None, None


def plot_topomap_familiarity_low_vs_high(log_fn=None, n_examples=3):
    """
    Graph 7–style: Topomaps comparing theta power for Low vs High familiarity examples.
    Row 0 = Low familiarity (1–2), Row 1 = High familiarity (5). Uses relative scaling (RdBu_r).
    """
    if not HAS_MNE:
        return
    behav = load_behavioural()
    low_examples = behav[behav["Familiarity_Group"] == "Low"].head(n_examples)
    high_examples = behav[behav["Familiarity_Group"] == "High"].head(n_examples)
    if len(low_examples) == 0 or len(high_examples) == 0:
        if log_fn:
            log_fn("Topomap Low vs High: need at least one example of each group. Skip.")
        return
    if log_fn:
        log_fn(f"Creating topomaps: Low familiarity n={len(low_examples)}, High n={len(high_examples)}")
    n_cols = max(len(low_examples), len(high_examples))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    # Row 0: Low familiarity
    for idx, (_, row) in enumerate(low_examples.iterrows()):
        sub, song_id = int(row["Subject"]), int(row["Song_ID"])
        info, theta_ch = _theta_channel_power_for_topomap(sub, song_id, log_fn=log_fn)
        if info is None or theta_ch is None:
            axes[0, idx].set_visible(False)
            continue
        # Friend-style: dB + relative to scalp mean
        theta_ch = np.asarray(theta_ch, dtype=float)
        eps = np.finfo(float).eps
        theta_db = 10 * np.log10(theta_ch + eps)
        theta_rel = theta_db - np.mean(theta_db)
        vmax = float(np.max(np.abs(theta_rel)))
        vlim = (-vmax, vmax) if vmax > 0 else (-1, 1)
        evoked_data = theta_rel[:, np.newaxis]
        evoked = EvokedArray(evoked_data, info, tmin=0)
        evoked.plot_topomap(
            times=[0],
            ch_type="eeg",
            show=False,
            axes=axes[0, idx],
            colorbar=False,
            cmap="RdBu_r",
            vlim=vlim,
            extrapolate="head",
        )
        axes[0, idx].set_title(f"Low familiarity\nSubj{sub}, Song{song_id}", fontsize=12, fontweight="bold")
    # Row 1: High familiarity
    for idx, (_, row) in enumerate(high_examples.iterrows()):
        sub, song_id = int(row["Subject"]), int(row["Song_ID"])
        info, theta_ch = _theta_channel_power_for_topomap(sub, song_id, log_fn=log_fn)
        if info is None or theta_ch is None:
            axes[1, idx].set_visible(False)
            continue
        theta_ch = np.asarray(theta_ch, dtype=float)
        eps = np.finfo(float).eps
        theta_db = 10 * np.log10(theta_ch + eps)
        theta_rel = theta_db - np.mean(theta_db)
        vmax = float(np.max(np.abs(theta_rel)))
        vlim = (-vmax, vmax) if vmax > 0 else (-1, 1)
        evoked_data = theta_rel[:, np.newaxis]
        evoked = EvokedArray(evoked_data, info, tmin=0)
        evoked.plot_topomap(
            times=[0],
            ch_type="eeg",
            show=False,
            axes=axes[1, idx],
            colorbar=False,
            cmap="RdBu_r",
            vlim=vlim,
            extrapolate="head",
        )
        axes[1, idx].set_title(f"High familiarity\nSubj{sub}, Song{song_id}", fontsize=12, fontweight="bold")
    # Hide unused subplots
    for idx in range(len(low_examples), n_cols):
        axes[0, idx].set_visible(False)
    for idx in range(len(high_examples), n_cols):
        axes[1, idx].set_visible(False)
    plt.suptitle("Theta Power Topography: Low vs High Familiarity (example trials)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / "explore_topomap_familiarity_low_vs_high.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    if log_fn:
        log_fn(f"Saved {out_path}")


def run_explore(max_runs=5, subject_id=1, song_id=1, make_all_plots=True):
    """
    Run exploratory pipeline: load one or a few runs, TFR, then plot.
    - If max_runs <= 0: use single (subject_id, song_id) for spectrogram and band-power plots.
    - If max_runs > 0: also build theta-by-familiarity table for that many runs and plot group comparison.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(msg):
        log_lines.append(msg)
        print(msg)

    log(f"Exploratory analysis started at {datetime.now().isoformat()}")
    log(f"Focus: Familiarity & Theta (alpha for comparison); methods adapted from enjoyment/alpha pipeline.")

    if not HAS_MNE:
        log("MNE not installed. Install with: pip install mne")
        return

    # Single-run TFR and band plots (example run)
    raw, events, event_id = load_eeg_with_events(subject_id, song_id)
    if raw is None:
        log(f"Could not load EEG for sub-{subject_id:03d} ses-{song_id:02d}. Check paths.")
        return
    log(f"Loaded sub-{subject_id:03d} ses-{song_id:02d}: {raw.times[-1]:.1f}s, {raw.info['sfreq']} Hz")

    raw = preprocess_raw(raw)
    epochs = create_music_epochs(raw, events, event_id, tmin=0.0, tmax=min(120.0, raw.times[-1] - 1))
    if epochs is None or len(epochs) == 0:
        log("No epochs created.")
        return
    log(f"Epochs: {len(epochs)}, tmin={epochs.tmin:.1f}s tmax={epochs.tmax:.1f}s")

    freqs = np.arange(2, 51, 2)
    power_tfr = compute_tfr(epochs, freqs=freqs, decim=4)
    log(f"TFR: freqs {power_tfr.freqs.min():.0f}-{power_tfr.freqs.max():.0f} Hz, times {power_tfr.times.min():.1f}-{power_tfr.times.max():.1f}s")

    electrodes_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=subject_id, ses=song_id, run=song_id)
    ch_to_region = get_electrode_regions(electrodes_tsv) if electrodes_tsv.exists() else {}
    frontal_chs = [c for c in power_tfr.ch_names if ch_to_region.get(c) == "frontal"]
    central_chs = [c for c in power_tfr.ch_names if ch_to_region.get(c) == "central"]
    parietal_chs = [c for c in power_tfr.ch_names if ch_to_region.get(c) == "parietal"]
    if not frontal_chs:
        frontal_chs = [power_tfr.ch_names[len(power_tfr.ch_names) // 2]]

    # —— Graph 1: Time–frequency spectrogram (one channel, similar to friend's) ——
    if make_all_plots:
        ch_idx = power_tfr.ch_names.index(frontal_chs[0]) if frontal_chs else 0
        ch_name = power_tfr.ch_names[ch_idx]
        fig_list = power_tfr.plot(
            [ch_idx], baseline=(None, 0), mode="logratio",
            title=f"Time–Frequency (theta 4–8 Hz) – {ch_name}",
            show=False,
        )
        fig = fig_list[0] if isinstance(fig_list, list) else fig_list
        fig.set_size_inches(12, 5)
        fig.savefig(OUT_DIR / "explore_spectrogram_theta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {OUT_DIR / 'explore_spectrogram_theta.png'}")

    # —— Graph 2a: All frequency bands over time (same as friend's "Frequency Band Power Over Time") ——
    if make_all_plots:
        band_colors = {
            "delta": "purple",
            "theta": "blue",
            "alpha": "green",
            "beta": "orange",
            "gamma": "red",
        }
        ch_idx = power_tfr.ch_names.index(frontal_chs[0]) if frontal_chs else 0
        ch_name = power_tfr.ch_names[ch_idx]
        fig, ax = plt.subplots(figsize=(14, 6))
        for band_name in FREQ_BANDS.keys():
            power_2d = extract_band_power(power_tfr, band_name, channels=frontal_chs)
            trace = np.mean(power_2d, axis=0)
            fmin, fmax = FREQ_BANDS[band_name]
            ax.plot(
                power_tfr.times, trace,
                label=f"{band_name.capitalize()} ({fmin}–{fmax} Hz)",
                color=band_colors[band_name],
                linewidth=2,
            )
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Power (dB)", fontsize=12)
        ax.set_title(f"Frequency Band Power Over Time – {ch_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "explore_all_bands_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {OUT_DIR / 'explore_all_bands_over_time.png'}")

    # —— Graph 2b: Theta & alpha only (focused) ——
    if make_all_plots:
        theta_power = extract_band_power(power_tfr, "theta", channels=frontal_chs)
        alpha_power = extract_band_power(power_tfr, "alpha", channels=frontal_chs)
        theta_trace = np.mean(theta_power, axis=0)
        alpha_trace = np.mean(alpha_power, axis=0)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(power_tfr.times, theta_trace, label="Theta (4–8 Hz)", color="blue", lw=2)
        ax.plot(power_tfr.times, alpha_trace, label="Alpha (8–13 Hz)", color="green", lw=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Theta & Alpha Power Over Time (frontal)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "explore_theta_alpha_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {OUT_DIR / 'explore_theta_alpha_over_time.png'}")

    # —— Graph 3: Theta power by region over time ——
    if make_all_plots and (frontal_chs or central_chs or parietal_chs):
        fig, ax = plt.subplots(figsize=(12, 5))
        for region_name, chs in [("Frontal", frontal_chs), ("Central", central_chs), ("Parietal", parietal_chs)]:
            if not chs:
                continue
            band = extract_band_power(power_tfr, "theta", channels=chs)
            trace = np.mean(band, axis=0)
            ax.plot(power_tfr.times, trace, label=f"Theta – {region_name}", lw=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Theta Power Over Time by Region")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "explore_theta_by_region_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {OUT_DIR / 'explore_theta_by_region_over_time.png'}")

    # —— Optional: theta vs familiarity group (aggregate over multiple runs) ——
    if max_runs > 0:
        behav = load_behavioural()
        behav = behav.head(max_runs)
        theta_rows = []
        for _, row in tqdm(behav.iterrows(), total=len(behav), desc="Runs"):
            sub, song_id = row["Subject"], row["Song_ID"]
            raw_r, events_r, event_id_r = load_eeg_with_events(sub, song_id)
            if raw_r is None:
                continue
            raw_r = preprocess_raw(raw_r)
            epochs_r = create_music_epochs(raw_r, events_r, event_id_r, tmin=0, tmax=min(120, raw_r.times[-1] - 1))
            if epochs_r is None or len(epochs_r) == 0:
                continue
            tfr_r = compute_tfr(epochs_r, freqs=np.arange(2, 51, 2), decim=8)
            elec_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
            ch_reg = get_electrode_regions(elec_tsv) if elec_tsv.exists() else {}
            fr_chs = [c for c in tfr_r.ch_names if ch_reg.get(c) == "frontal"]
            if not fr_chs:
                fr_chs = [tfr_r.ch_names[len(tfr_r.ch_names) // 2]]
            theta_p = extract_band_power(tfr_r, "theta", channels=fr_chs)
            theta_rows.append({**row.to_dict(), "theta_frontal": np.mean(theta_p)})
        if theta_rows:
            df = pd.DataFrame(theta_rows)
            low = df.loc[df["Familiarity_Group"] == "Low", "theta_frontal"].dropna()
            high = df.loc[df["Familiarity_Group"] == "High", "theta_frontal"].dropna()
            if len(low) >= 1 or len(high) >= 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                boxes, labels = [], []
                if len(low) >= 1:
                    boxes.append(low)
                    labels.append("Low")
                if len(high) >= 1:
                    boxes.append(high)
                    labels.append("High")
                ax.boxplot(boxes, tick_labels=labels, patch_artist=True)
                ax.set_xlabel("Familiarity Group")
                ax.set_ylabel("Theta power (frontal)")
                ax.set_title("Exploratory: Theta vs Familiarity Group")
                fig.tight_layout()
                fig.savefig(OUT_DIR / "explore_theta_vs_familiarity_group.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                log(f"Saved {OUT_DIR / 'explore_theta_vs_familiarity_group.png'} (n={len(df)})")

    # —— Graph 7–style: Topomaps comparing Low vs High familiarity (example trials) ——
    if make_all_plots:
        plot_topomap_familiarity_low_vs_high(log_fn=log, n_examples=3)

    # Save log
    log_path = OUT_DIR / f"explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exploratory: Familiarity & Theta (event-based TFR, like enjoyment/alpha).")
    p.add_argument("--max-runs", type=int, default=0, help="If >0, also aggregate theta over this many runs and plot vs familiarity.")
    p.add_argument("--sub", type=int, default=1, help="Subject ID for single-run spectrogram/band plots.")
    p.add_argument("--ses", type=int, default=1, help="Session/song ID for single-run plots.")
    p.add_argument("--no-plots", action="store_true", help="Skip generating plots (e.g. TFR only).")
    args = p.parse_args()
    run_explore(max_runs=args.max_runs, subject_id=args.sub, song_id=args.ses, make_all_plots=not args.no_plots)
