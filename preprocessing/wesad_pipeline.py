"""
WESAD preprocessing pipeline.

For each subject S{n}:
  1. Load S{n}.pkl — contains chest (700 Hz) and wrist (64 Hz) signals + label array
  2. Use label 1 (baseline) as normal, label 2 (stress) as anomaly; discard others
  3. Extract contiguous baseline and stress segments (min 10 sec each)
  4. ECG  (chest, 700 Hz → 100 Hz): resample, bandpass filter (0.5–40 Hz)
  5. BVP  (wrist, 64 Hz → 100 Hz):  resample, bandpass filter (0.5–5 Hz)
  6. Fit StandardScaler on baseline condition, apply to stress
  7. Save to data/processed/wesad_S{n}/:
       ecg_baseline.npy  ecg_stress.npy  ecg_scaler.pkl
       ppg_baseline.npy  ppg_stress.npy  ppg_scaler.pkl

Usage:
    python -m preprocessing.wesad_pipeline
    python -m preprocessing.wesad_pipeline --subjects S2 S3 --overwrite
    python -m preprocessing.wesad_pipeline --source /path/to/WESAD
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler

REPO_ROOT   = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "processed"

CHEST_FS = 700.0
WRIST_FS =  64.0
TARGET_FS = 100.0

LABEL_BASELINE = 1
LABEL_STRESS   = 2


def _load_pkl(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _downsample_labels(labels_chest: np.ndarray, n_wrist: int) -> np.ndarray:
    """Map chest-rate label array (700 Hz) down to wrist sample count (64 Hz)."""
    indices = np.minimum(
        (np.arange(n_wrist) * (CHEST_FS / WRIST_FS)).astype(int),
        len(labels_chest) - 1,
    )
    return labels_chest[indices]


def _extract_segments(signal: np.ndarray, labels: np.ndarray, label_val: int, min_len: int) -> list[np.ndarray]:
    """Return contiguous runs where labels == label_val with length >= min_len."""
    segments, in_seg, start = [], False, 0
    for i in range(len(labels)):
        if labels[i] == label_val and not in_seg:
            start, in_seg = i, True
        elif labels[i] != label_val and in_seg:
            if i - start >= min_len:
                segments.append(signal[start:i])
            in_seg = False
    if in_seg and len(labels) - start >= min_len:
        segments.append(signal[start:])
    return segments


def _process_segments(
    segs: list[np.ndarray],
    source_fs: float,
    bandpass_low: float,
    bandpass_high: float,
) -> np.ndarray:
    """
    Resample each segment from source_fs to TARGET_FS, bandpass filter,
    concatenate, and return float32 array [T, C].
    """
    nyq = TARGET_FS / 2.0
    high = min(bandpass_high, nyq - 1.0)
    b, a = butter(4, [bandpass_low / nyq, high / nyq], btype="band")

    out = []
    for seg in segs:
        # seg shape: [T] or [T, C]
        if seg.ndim == 1:
            seg = seg[:, None]
        n_out = int(round(seg.shape[0] * TARGET_FS / source_fs))
        channels = []
        for c in range(seg.shape[1]):
            r = resample(seg[:, c], n_out).astype(np.float32)
            r = filtfilt(b, a, r).astype(np.float32)
            channels.append(r)
        out.append(np.stack(channels, axis=1))
    return np.concatenate(out, axis=0)


def process_subject(
    subject_id: str,
    source_dir: Path,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> bool:
    """
    Preprocess one WESAD subject.

    Returns True if processed, False if skipped.
    """
    out_dir = output_root / f"wesad_{subject_id}"

    if not overwrite and (out_dir / "ecg_scaler.pkl").exists():
        print(f"  [{subject_id}] already processed — skipping (use --overwrite to redo)")
        return False

    pkl_path = source_dir / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        print(f"  [{subject_id}] {pkl_path} not found — skipping")
        return False

    data    = _load_pkl(pkl_path)
    labels  = data["label"].astype(int)          # [T_chest], 700 Hz
    chest   = data["signal"]["chest"]
    wrist   = data["signal"]["wrist"]

    ecg_raw = chest["ECG"].astype(np.float32)    # [T_chest, 1]
    bvp_raw = wrist["BVP"].astype(np.float32)    # [T_wrist, 1]

    label_bvp = _downsample_labels(labels, len(bvp_raw))

    min_ecg = int(10.0 * CHEST_FS)
    min_bvp = int(10.0 * WRIST_FS)

    ecg_baseline_segs = _extract_segments(ecg_raw, labels,    LABEL_BASELINE, min_ecg)
    ecg_stress_segs   = _extract_segments(ecg_raw, labels,    LABEL_STRESS,   min_ecg)
    bvp_baseline_segs = _extract_segments(bvp_raw, label_bvp, LABEL_BASELINE, min_bvp)
    bvp_stress_segs   = _extract_segments(bvp_raw, label_bvp, LABEL_STRESS,   min_bvp)

    if not ecg_baseline_segs or not ecg_stress_segs:
        print(f"  [{subject_id}] missing baseline or stress ECG segments — skipping")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- ECG ---
    ecg_base = _process_segments(ecg_baseline_segs, CHEST_FS, bandpass_low=0.5, bandpass_high=40.0)
    ecg_strs = _process_segments(ecg_stress_segs,   CHEST_FS, bandpass_low=0.5, bandpass_high=40.0)

    ecg_scaler = StandardScaler()
    ecg_scaler.fit(ecg_base)
    ecg_base = ecg_scaler.transform(ecg_base).astype(np.float32)
    ecg_strs = ecg_scaler.transform(ecg_strs).astype(np.float32)

    np.save(out_dir / "ecg_baseline.npy", ecg_base)
    np.save(out_dir / "ecg_stress.npy",   ecg_strs)
    with open(out_dir / "ecg_scaler.pkl", "wb") as f:
        pickle.dump(ecg_scaler, f)
    print(f"  [{subject_id}] ecg_baseline.npy  shape={ecg_base.shape}")
    print(f"  [{subject_id}] ecg_stress.npy    shape={ecg_strs.shape}")

    # --- PPG (BVP) ---
    if bvp_baseline_segs and bvp_stress_segs:
        ppg_base = _process_segments(bvp_baseline_segs, WRIST_FS, bandpass_low=0.5, bandpass_high=5.0)
        ppg_strs = _process_segments(bvp_stress_segs,   WRIST_FS, bandpass_low=0.5, bandpass_high=5.0)

        ppg_scaler = StandardScaler()
        ppg_scaler.fit(ppg_base)
        ppg_base = ppg_scaler.transform(ppg_base).astype(np.float32)
        ppg_strs = ppg_scaler.transform(ppg_strs).astype(np.float32)

        np.save(out_dir / "ppg_baseline.npy", ppg_base)
        np.save(out_dir / "ppg_stress.npy",   ppg_strs)
        with open(out_dir / "ppg_scaler.pkl", "wb") as f:
            pickle.dump(ppg_scaler, f)
        print(f"  [{subject_id}] ppg_baseline.npy  shape={ppg_base.shape}")
        print(f"  [{subject_id}] ppg_stress.npy    shape={ppg_strs.shape}")
    else:
        print(f"  [{subject_id}] skipping PPG — insufficient baseline or stress BVP segments")

    return True


def process_all(
    source_dir: Path,
    output_root: Path = OUTPUT_ROOT,
    subjects: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Process all WESAD subjects in source_dir (or a specific subset)."""
    if subjects:
        subject_ids = subjects
    else:
        subject_ids = sorted(
            p.name for p in source_dir.iterdir()
            if p.is_dir() and p.name.startswith("S") and p.name[1:].isdigit()
        )

    if not subject_ids:
        raise FileNotFoundError(f"No WESAD subject directories found in {source_dir}")

    print(f"Processing {len(subject_ids)} WESAD subject(s) → {output_root}\n")
    processed = 0
    for sid in subject_ids:
        ok = process_subject(sid, source_dir, output_root, overwrite=overwrite)
        if ok:
            processed += 1

    print(f"\nDone. {processed}/{len(subject_ids)} subjects processed.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess WESAD dataset.")
    p.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent.parent.parent / "WESAD"),
        help="Path to WESAD directory containing S2/, S3/, ... subject folders",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help="Root directory for processed output",
    )
    p.add_argument(
        "--subjects", nargs="+", default=None,
        help="Specific subject IDs to process (e.g. S2 S3); default = all",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_all(
        source_dir=Path(args.source),
        output_root=Path(args.output),
        subjects=args.subjects,
        overwrite=args.overwrite,
    )
