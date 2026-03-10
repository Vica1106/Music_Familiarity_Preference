#!/usr/bin/env python3
"""
Music Familiarity vs Theta-band EEG Analysis

Research question: Does music familiarity (1-5 scale) affect theta-band EEG activity?
- Preprocessing: Keep only Familiarity in {1, 2, 5}. Group: Low (1,2) vs High (5).
- Theta power (4-8 Hz) by brain region: frontal, central, parietal.
- Statistics: t-test and Mann-Whitney U (Low vs High).
- Main figure: Boxplot Theta Power vs Familiarity Group.

Quick test on small data (after installing requirements):
  python analysis_theta_familiarity.py --max-runs 5
"""

import os
import re
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Optional: MNE for EEGLAB .set, PSD, and topomap
try:
    import mne
    from mne.io import read_raw_eeglab
    from mne.filter import filter_data
    from mne.channels import make_dig_montage
    from mne.viz import plot_topomap
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Logger: configured in main() to write to console + file
logger = logging.getLogger("theta_familiarity")

# Paths
DATA_ROOT = Path(__file__).resolve().parent / "data"
BEHAV_PATH = DATA_ROOT / "stimuli" / "Behavioural_data"
# String templates (Path has no .format()); use DATA_ROOT / template.format(...)
ELECTRODES_TEMPLATE = "sub-{sub:03d}/ses-{ses:02d}/eeg/sub-{sub:03d}_ses-{ses:02d}_task-MusicListening_run-{run:d}_electrodes.tsv"
EEG_TEMPLATE = "sub-{sub:03d}/ses-{ses:02d}/eeg/sub-{sub:03d}_ses-{ses:02d}_task-MusicListening_run-{run:d}_eeg.set"
THETA_LOW, THETA_HIGH = 4.0, 8.0
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_CSV = OUT_DIR / "theta_by_familiarity.csv"
OUT_FIG = OUT_DIR / "theta_power_vs_familiarity_group.png"
OUT_FIG_REGIONS = OUT_DIR / "theta_power_by_region.png"
OUT_FIG_TOPO = OUT_DIR / "theta_topomap.png"
OUT_FIG_TOPO_GROUP = OUT_DIR / "theta_topomap_by_familiarity_group.png"
OUT_CH_CSV = OUT_DIR / "theta_channels_by_trial.csv"
LOG_FILE = "analysis_{}.log"  # timestamp filled in main()


def load_behavioural():
    """Load behavioural data; keep Familiarity in {1,2,5}; add Group Low/High."""
    rows = []
    with open(BEHAV_PATH) as f:
        header = f.readline()
        for line in f:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 4:
                continue
            subject = int(parts[0])
            song_id = int(parts[1])
            enjoyment = int(parts[2])
            familiarity = int(parts[3])
            if familiarity not in (1, 2, 5):
                continue
            group = "Low" if familiarity in (1, 2) else "High"
            rows.append({
                "Subject": subject,
                "Song_ID": song_id,
                "Enjoyment": enjoyment,
                "Familiarity": familiarity,
                "Familiarity_Group": group,
            })
    return pd.DataFrame(rows)


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
            # EGI coords typically in cm -> convert to meters for MNE
            ch_pos[name] = np.array([x, y, z]) / 100.0
    return make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def make_info_from_electrodes(electrodes_tsv, sfreq=1000.0):
    """Create an MNE Info with montage for topomap plotting (no need to read EEG)."""
    if not HAS_MNE:
        return None
    montage = make_montage_from_electrodes(electrodes_tsv)
    info = mne.create_info(ch_names=montage.ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    return info


def find_first_existing_electrodes(behav_df):
    """Find a electrodes.tsv path that exists, using rows in behav_df."""
    for _, row in behav_df.iterrows():
        sub, song_id = int(row["Subject"]), int(row["Song_ID"])
        electrodes_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
        if electrodes_tsv.exists():
            return electrodes_tsv
    return None


def get_electrode_regions(electrodes_tsv):
    """Build channel name -> region (frontal/central/parietal) from electrodes.tsv z."""
    ch_to_region = {}
    with open(electrodes_tsv) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            if name == "E129":
                continue  # reference
            if z >= 5:
                ch_to_region[name] = "frontal"
            elif z >= -2:
                ch_to_region[name] = "central"
            else:
                ch_to_region[name] = "parietal"
    return ch_to_region


def compute_theta_power_per_channel(data_2d, sfreq):
    """Compute mean theta (4-8 Hz) power per channel. data_2d: (n_channels, n_times)."""
    if not HAS_MNE:
        # Fallback: simple bandpass via FFT (no MNE)
        n_ch, n_times = data_2d.shape
        n_fft = min(2 ** int(np.ceil(np.log2(n_times))), n_times)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
        theta_mask = (freqs >= THETA_LOW) & (freqs <= THETA_HIGH)
        power_per_ch = []
        for ch in range(n_ch):
            x = data_2d[ch].astype(np.float64)
            x = x - np.mean(x)
            X = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
            theta_power = np.mean(X[theta_mask])
            power_per_ch.append(theta_power)
        return np.array(power_per_ch)
    # With MNE: bandpass then variance as power proxy
    from mne.filter import filter_data
    filtered = filter_data(
        data_2d, sfreq, THETA_LOW, THETA_HIGH, verbose=False
    )
    return np.mean(filtered ** 2, axis=1)


def compute_theta_by_region(eeg_path, electrodes_tsv):
    """Load one EEG run, compute mean theta power per region. Returns dict theta_frontal, theta_central, theta_parietal."""
    import warnings
    ch_to_region = get_electrode_regions(electrodes_tsv)
    if HAS_MNE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # annotations outside range, boundary events
            raw = read_raw_eeglab(eeg_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        # Use only channels we have regions for (E1-E128)
        ch_names = [c for c in raw.ch_names if c in ch_to_region]
        raw.pick(ch_names)
        data, _ = raw.get_data(return_times=True)  # (n_ch, n_times)
    else:
        # Minimal reader for .set: would need scipy.io.loadmat or eeglabio
        raise FileNotFoundError("MNE is required to read EEGLAB .set files. Install: pip install mne")
    power_per_ch = compute_theta_power_per_channel(data, sfreq)
    # Average by region
    frontal_chs = [c for c in ch_names if ch_to_region[c] == "frontal"]
    central_chs = [c for c in ch_names if ch_to_region[c] == "central"]
    parietal_chs = [c for c in ch_names if ch_to_region[c] == "parietal"]
    ch_to_idx = {c: i for i, c in enumerate(ch_names)}
    def mean_power(ch_list):
        if not ch_list:
            return np.nan
        return np.mean([power_per_ch[ch_to_idx[c]] for c in ch_list])
    return {
        "theta_frontal": mean_power(frontal_chs),
        "theta_central": mean_power(central_chs),
        "theta_parietal": mean_power(parietal_chs),
    }


def compute_theta_per_channel_for_topomap(eeg_path, electrodes_tsv):
    """
    Load one run, set montage, compute theta power per channel. For topography.
    Returns (info, ch_names, theta_per_ch) or (None, None, None).
    """
    import warnings
    if not HAS_MNE:
        return None, None, None
    try:
        montage = make_montage_from_electrodes(electrodes_tsv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = read_raw_eeglab(eeg_path, preload=True, verbose=False)
        ch_names = [c for c in raw.ch_names if c in montage.ch_names]
        if len(ch_names) < 10:
            return None, None, None
        raw.pick(ch_names)
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        data, _ = raw.get_data(return_times=True)
        power_per_ch = compute_theta_power_per_channel(data, raw.info["sfreq"])
        return raw.info, ch_names, power_per_ch
    except Exception:
        return None, None, None


def load_or_build_theta_channels_cache(behav_df):
    """
    Cache per-channel theta for each trial/run.
    Returns a DataFrame with columns: Subject, Song_ID, Familiarity, Familiarity_Group, channel, theta_power.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_CH_CSV.exists():
        try:
            df_ch = pd.read_csv(OUT_CH_CSV)
            if {"channel", "theta_power", "Familiarity_Group"}.issubset(df_ch.columns):
                logger.info("Found cached channel-level theta: %s", OUT_CH_CSV)
                return df_ch
        except Exception as e:
            logger.warning("Failed reading %s: %s (will recompute)", OUT_CH_CSV, e)

    logger.info("Building channel-level theta cache (for topomaps)...")
    rows = []
    for _, row in tqdm(behav_df.iterrows(), total=len(behav_df), desc="Topomap runs", unit="run"):
        sub, song_id = int(row["Subject"]), int(row["Song_ID"])
        electrodes_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
        eeg_path = DATA_ROOT / EEG_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
        if not eeg_path.exists() or not electrodes_tsv.exists():
            continue
        info, ch_names, theta_ch = compute_theta_per_channel_for_topomap(eeg_path, electrodes_tsv)
        if info is None or theta_ch is None or ch_names is None:
            continue
        for ch, val in zip(ch_names, theta_ch):
            rows.append({
                "Subject": sub,
                "Song_ID": song_id,
                "Familiarity": int(row["Familiarity"]),
                "Familiarity_Group": row["Familiarity_Group"],
                "channel": ch,
                "theta_power": float(val),
            })
    df_ch = pd.DataFrame(rows)
    df_ch.to_csv(OUT_CH_CSV, index=False)
    logger.info("Saved %s", OUT_CH_CSV)
    return df_ch


def build_theta_table(behav):
    """For each row in behav, load corresponding EEG and fill theta by region."""
    rows = []
    n = len(behav)
    for idx, row in tqdm(behav.iterrows(), total=n, desc="EEG runs", unit="run"):
        sub = row["Subject"]
        song_id = row["Song_ID"]
        electrodes_tsv = DATA_ROOT / ELECTRODES_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
        eeg_path = DATA_ROOT / EEG_TEMPLATE.format(sub=sub, ses=song_id, run=song_id)
        if not eeg_path.exists() or not electrodes_tsv.exists():
            logger.warning("Skip sub-%03d ses-%02d (file missing)", sub, song_id)
            continue
        try:
            theta = compute_theta_by_region(eeg_path, electrodes_tsv)
        except Exception as e:
            logger.warning("Error sub-%03d ses-%02d: %s", sub, song_id, e)
            continue
        rows.append({
            **row.to_dict(),
            **theta,
        })
    return pd.DataFrame(rows)


def run_statistics(df):
    """t-test and Mann-Whitney U for Low vs High (each theta region)."""
    low = df[df["Familiarity_Group"] == "Low"]
    high = df[df["Familiarity_Group"] == "High"]
    for region in ["theta_frontal", "theta_central", "theta_parietal"]:
        a, b = low[region].dropna(), high[region].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p_t = stats.ttest_ind(a, b)
        u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        logger.info("")
        logger.info("%s:", region)
        logger.info("  Low  n=%d mean=%.4f std=%.4f", len(a), a.mean(), a.std())
        logger.info("  High n=%d mean=%.4f std=%.4f", len(b), b.mean(), b.std())
        logger.info("  t-test: t=%.4f p=%.4f", t, p_t)
        logger.info("  Mann-Whitney U: U=%.1f p=%.4f", u, p_mw)


def plot_boxplot_main(df, out_path=OUT_FIG):
    """Main report figure: Boxplot Theta Power (frontal) vs Familiarity Group."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    low = df.loc[df["Familiarity_Group"] == "Low", "theta_frontal"].dropna()
    high = df.loc[df["Familiarity_Group"] == "High", "theta_frontal"].dropna()
    ax.boxplot([low, high], tick_labels=["Low", "High"], patch_artist=True)
    ax.set_xlabel("Familiarity Group")
    ax.set_ylabel("Theta power (4–8 Hz, frontal)")
    ax.set_title("Theta Power vs Familiarity Group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_boxplot_by_region(df, out_path=OUT_FIG_REGIONS):
    """Three panels: theta_frontal, theta_central, theta_parietal vs Familiarity Group."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, col in zip(axes, ["theta_frontal", "theta_central", "theta_parietal"]):
        low = df.loc[df["Familiarity_Group"] == "Low", col].dropna()
        high = df.loc[df["Familiarity_Group"] == "High", col].dropna()
        ax.boxplot([low, high], tick_labels=["Low", "High"], patch_artist=True)
        ax.set_xlabel("Familiarity Group")
        ax.set_ylabel("Theta power (4–8 Hz)")
        ax.set_title(col.replace("theta_", "").capitalize())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_theta_topomap(info, data_1d, title, out_path, ch_names=None):
    """Plot brain topography for theta power. data_1d: (n_channels,) in same order as info.ch_names."""
    if not HAS_MNE or info is None or data_1d is None:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure order matches info
    if ch_names is not None:
        order = [info["ch_names"].index(c) for c in ch_names if c in info["ch_names"]]
        data_1d = np.asarray(data_1d)[order]
    # Friend-style: convert to dB and center by scalp mean, use symmetric vlim and RdBu_r
    data_1d = np.asarray(data_1d, dtype=float)
    eps = np.finfo(float).eps
    data_db = 10 * np.log10(data_1d + eps)
    rel = data_db - np.mean(data_db)
    vmax = float(np.max(np.abs(rel))) or 1.0
    vlim = (-vmax, vmax)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    try:
        plot_topomap(
            rel,
            info,
            axes=ax,
            show=False,
            contours=0,
            extrapolate="head",
            cmap="RdBu_r",
            vlim=vlim,
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_path)
    except Exception as e:
        logger.warning("Topomap plot failed: %s", e)
        plt.close("all")


def build_and_plot_theta_topomap(behav, df_with_groups):
    """
    Compute mean theta power per channel (overall and by Low/High), plot topomaps.
    Uses same runs as in df_with_groups; needs to re-load EEG to get per-channel power.
    """
    if not HAS_MNE:
        return
    # Load or build channel-level cache first
    df_ch = load_or_build_theta_channels_cache(behav)
    if df_ch is None or len(df_ch) == 0:
        logger.warning("No channel-level theta available for topomap.")
        return

    # Build an Info with montage without re-reading EEG
    electrodes_tsv = find_first_existing_electrodes(behav)
    if electrodes_tsv is None:
        logger.warning("No electrodes.tsv found for topomap montage.")
        return
    ref_info = make_info_from_electrodes(electrodes_tsv)
    if ref_info is None:
        return
    ref_ch = list(ref_info["ch_names"])

    # Helper: get mean theta per channel for a subset
    def mean_by_channel(sub_df):
        m = sub_df.groupby("channel")["theta_power"].mean()
        return np.array([m.get(ch, np.nan) for ch in ref_ch], dtype=float)

    overall = mean_by_channel(df_ch)
    if np.isfinite(overall).any():
        plot_theta_topomap(ref_info, overall, "Theta power (4–8 Hz) – average over trials", OUT_FIG_TOPO, ch_names=ref_ch)

    low_df = df_ch[df_ch["Familiarity_Group"] == "Low"]
    high_df = df_ch[df_ch["Familiarity_Group"] == "High"]
    if len(low_df) == 0 or len(high_df) == 0:
        logger.warning("Topomap Low vs High requires both groups (got Low=%d, High=%d).", len(low_df), len(high_df))
        return

    low_theta = mean_by_channel(low_df)
    high_theta = mean_by_channel(high_df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        # Friend-style dB + relative scaling for each map
        eps = np.finfo(float).eps
        low_db = 10 * np.log10(low_theta + eps)
        low_rel = low_db - np.mean(low_db)
        high_db = 10 * np.log10(high_theta + eps)
        high_rel = high_db - np.mean(high_db)
        vmax = float(np.max(np.abs(np.concatenate([low_rel, high_rel])))) or 1.0
        vlim = (-vmax, vmax)
        plot_topomap(
            low_rel,
            ref_info,
            axes=axes[0],
            show=False,
            contours=0,
            extrapolate="head",
            cmap="RdBu_r",
            vlim=vlim,
        )
        axes[0].set_title("Theta – Low familiarity")
        plot_topomap(
            high_rel,
            ref_info,
            axes=axes[1],
            show=False,
            contours=0,
            extrapolate="head",
            cmap="RdBu_r",
            vlim=vlim,
        )
        axes[1].set_title("Theta – High familiarity")
        plt.tight_layout()
        plt.savefig(OUT_FIG_TOPO_GROUP, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", OUT_FIG_TOPO_GROUP)
    except Exception as e:
        logger.warning("Topomap by-group figure failed: %s", e)
        plt.close("all")


def setup_logging(log_path):
    """Configure logger to write to both console and file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fmt_console = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt_console)
    logger.addHandler(fh)
    logger.addHandler(sh)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Theta vs Familiarity analysis")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Only process first N runs (for quick test on small data). Example: --max-runs 5",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached CSVs and recompute EEG-derived features.",
    )
    args = parser.parse_args()

    # Log to results/analysis_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / LOG_FILE.format(timestamp)  # e.g. results/analysis_20250309_123456.log
    setup_logging(log_path)
    logger.info("Log file: %s", log_path)

    logger.info("Loading behavioural data (Familiarity 1, 2, 5 only)...")
    behav = load_behavioural()
    if args.max_runs is not None:
        behav = behav.head(args.max_runs)
        logger.info("[Quick test] Using only first %d runs.", args.max_runs)
    logger.info("Trials: %d (Low: %d, High: %d)",
                len(behav), (behav["Familiarity_Group"] == "Low").sum(), (behav["Familiarity_Group"] == "High").sum())

    if OUT_CSV.exists() and not args.recompute:
        logger.info("Found cached features: %s", OUT_CSV)
        df = pd.read_csv(OUT_CSV)
    elif not HAS_MNE:
        logger.info("MNE not installed. Creating a dummy theta table from behavioural data only.")
        logger.info("Install MNE and run again to compute real theta: pip install mne")
        df = behav.copy()
        np.random.seed(42)
        n = len(df)
        df["theta_frontal"] = np.random.randn(n).cumsum() + 10
        df["theta_central"] = np.random.randn(n).cumsum() + 10
        df["theta_parietal"] = np.random.randn(n).cumsum() + 10
    else:
        logger.info("Computing theta power per run (this may take a few minutes)...")
        df = build_theta_table(behav)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        logger.info("Saved %s", OUT_CSV)

    # Ensure output directory exists even when loading cached CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("--- Statistics (Low vs High) ---")
    run_statistics(df)

    logger.info("")
    logger.info("--- Figures ---")
    plot_boxplot_main(df)
    plot_boxplot_by_region(df)
    if HAS_MNE and len(df) > 0:
        logger.info("Computing theta topomaps...")
        build_and_plot_theta_topomap(behav, df)

    logger.info("")
    logger.info("Done. Log saved to %s", log_path)


if __name__ == "__main__":
    main()
