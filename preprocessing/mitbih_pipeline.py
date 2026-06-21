"""
MIT-BIH Arrhythmia Database preprocessing pipeline.

For each recording:
  1. Load MLII channel from {record}.csv (360 Hz)
  2. Parse beat annotations from {record}annotations.txt
  3. Dilate arrhythmia beat positions by ±0.5 sec to create an anomaly mask
  4. Extract contiguous normal and arrhythmia segments (min 2 sec each)
  5. Resample to target_fs (default 100 Hz), bandpass filter, standardize
  6. Save to data/processed/mitbih_{record}/ecg_normal.npy + ecg_arrhythmia.npy

Usage:
    python -m preprocessing.mitbih_pipeline
    python -m preprocessing.mitbih_pipeline --records 100 105 --overwrite
    python -m preprocessing.mitbih_pipeline --source /path/to/MIT-BIH
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler

REPO_ROOT    = Path(__file__).resolve().parent.parent
MIT_BIH_FS   = 360.0
OUTPUT_ROOT  = REPO_ROOT / "data" / "processed"

# Rhythm/signal quality markers that are not actual beats
NON_BEAT_TYPES = {"+", "~", "|", '"', "^", "[", "]", "!", "x", "U", "M"}

# The only beat type we treat as normal; everything else is arrhythmia
NORMAL_BEAT_TYPES = {"N"}


def _load_signal(csv_path: Path) -> np.ndarray:
    """
    Load the primary ECG channel from a MIT-BIH CSV.

    Prefers MLII; falls back to the first non-sample-index column for the
    handful of recordings (102, 104) that only have V-lead channels.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip("'\" ") for c in df.columns]
    col = "MLII" if "MLII" in df.columns else [c for c in df.columns if c != "sample #"][0]
    return df[col].values.astype(np.float32)


def _load_annotations(ann_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a MIT-BIH annotations file.

    Returns:
        sample_indices: int array of beat sample positions
        beat_types:     str array of annotation labels
    """
    samples, types = [], []
    with open(ann_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                beat_type = parts[2]
                if beat_type in NON_BEAT_TYPES:
                    continue
                samples.append(int(parts[1]))
                types.append(beat_type)
            except (ValueError, IndexError):
                continue
    return np.array(samples, dtype=int), np.array(types, dtype=str)


def _make_anomaly_mask(n: int, arrhythmia_samples: np.ndarray, context: int) -> np.ndarray:
    """Boolean mask of length n; True = within context samples of an arrhythmia beat."""
    mask = np.zeros(n, dtype=bool)
    for s in arrhythmia_samples:
        lo = max(0, s - context)
        hi = min(n, s + context + 1)
        mask[lo:hi] = True
    return mask


def _extract_segments(signal: np.ndarray, mask: np.ndarray, target: bool, min_len: int) -> list[np.ndarray]:
    """Return contiguous runs where mask == target with length >= min_len."""
    segments, in_seg, start = [], False, 0
    for i in range(len(mask)):
        if mask[i] == target and not in_seg:
            start, in_seg = i, True
        elif mask[i] != target and in_seg:
            if i - start >= min_len:
                segments.append(signal[start:i])
            in_seg = False
    if in_seg and len(mask) - start >= min_len:
        segments.append(signal[start:])
    return segments


def _process_segments(
    segs: list[np.ndarray],
    target_fs: float,
    bandpass_low: float,
    bandpass_high: float,
) -> np.ndarray:
    """Resample each segment to target_fs, bandpass filter, concatenate → [T, 1]."""
    nyq = target_fs / 2.0
    high = min(bandpass_high, nyq - 1.0)
    b, a = butter(4, [bandpass_low / nyq, high / nyq], btype="band")

    out = []
    for seg in segs:
        n_out = int(round(len(seg) * target_fs / MIT_BIH_FS))
        r = resample(seg, n_out).astype(np.float32)
        r = filtfilt(b, a, r).astype(np.float32)
        out.append(r)
    return np.concatenate(out).reshape(-1, 1)


def process_record(
    record_id: str,
    source_dir: Path,
    output_root: Path = OUTPUT_ROOT,
    target_fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    context_sec: float = 0.5,
    min_normal_sec: float = 2.0,
    min_arrhy_sec: float = 0.5,
    overwrite: bool = False,
) -> bool:
    """
    Preprocess one MIT-BIH recording.

    Returns True if processed, False if skipped.
    """
    out_dir = output_root / f"mitbih_{record_id}"

    if not overwrite and (out_dir / "ecg_scaler.pkl").exists():
        print(f"  [{record_id}] already processed — skipping (use --overwrite to redo)")
        return False

    csv_path = source_dir / f"{record_id}.csv"
    ann_path = source_dir / f"{record_id}annotations.txt"
    if not csv_path.exists() or not ann_path.exists():
        print(f"  [{record_id}] missing CSV or annotations — skipping")
        return False

    signal = _load_signal(csv_path)
    sample_indices, beat_types = _load_annotations(ann_path)

    arrhythmia_samples = sample_indices[~np.isin(beat_types, list(NORMAL_BEAT_TYPES))]
    context     = int(context_sec    * MIT_BIH_FS)
    min_normal  = int(min_normal_sec * MIT_BIH_FS)
    min_arrhy   = int(min_arrhy_sec  * MIT_BIH_FS)

    mask         = _make_anomaly_mask(len(signal), arrhythmia_samples, context)
    normal_segs  = _extract_segments(signal, mask, target=False, min_len=min_normal)
    arrhy_segs   = _extract_segments(signal, mask, target=True,  min_len=min_arrhy)

    if not normal_segs:
        print(f"  [{record_id}] no normal segments found — skipping")
        return False

    normal_arr = _process_segments(normal_segs, target_fs, bandpass_low, bandpass_high)

    scaler = StandardScaler()
    scaler.fit(normal_arr)
    normal_arr = scaler.transform(normal_arr).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ecg_normal.npy", normal_arr)
    print(f"  [{record_id}] ecg_normal.npy       shape={normal_arr.shape}")

    if arrhy_segs:
        arrhy_arr = _process_segments(arrhy_segs, target_fs, bandpass_low, bandpass_high)
        arrhy_arr = scaler.transform(arrhy_arr).astype(np.float32)
        np.save(out_dir / "ecg_arrhythmia.npy", arrhy_arr)
        print(f"  [{record_id}] ecg_arrhythmia.npy  shape={arrhy_arr.shape}")
    else:
        print(f"  [{record_id}] no arrhythmia segments (normal-only recording)")

    with open(out_dir / "ecg_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return True


def process_all(
    source_dir: Path,
    output_root: Path = OUTPUT_ROOT,
    records: list[str] | None = None,
    target_fs: float = 100.0,
    overwrite: bool = False,
) -> None:
    """Process all MIT-BIH recordings in source_dir (or a specific subset)."""
    if records:
        record_ids = records
    else:
        record_ids = sorted(
            p.stem for p in source_dir.glob("*.csv")
            if p.stem.isdigit()
        )

    if not record_ids:
        raise FileNotFoundError(f"No MIT-BIH CSVs found in {source_dir}")

    print(f"Processing {len(record_ids)} MIT-BIH recording(s) → {output_root}\n")
    processed = 0
    for rid in record_ids:
        ok = process_record(rid, source_dir, output_root, target_fs=target_fs, overwrite=overwrite)
        if ok:
            processed += 1

    print(f"\nDone. {processed}/{len(record_ids)} recordings processed.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess MIT-BIH Arrhythmia Database.")
    p.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent.parent.parent / "MIT-BIH" / "mitbih_database"),
        help="Path to MIT-BIH directory containing .csv and annotations.txt files",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help="Root directory for processed output",
    )
    p.add_argument(
        "--records", nargs="+", default=None,
        help="Specific recording IDs to process (e.g. 100 105); default = all",
    )
    p.add_argument("--target-fs", type=float, default=100.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_all(
        source_dir=Path(args.source),
        output_root=Path(args.output),
        records=args.records,
        target_fs=args.target_fs,
        overwrite=args.overwrite,
    )
