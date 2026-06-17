"""
Batch preprocessing driver.

Reads data/manifests/sessions.csv and runs the appropriate pipeline
(ECG or PPG) for each session. Within each patient, the resting condition
is always processed first so its scaler can be reused for all other conditions.

Usage:
    python preprocessing/run_all.py
    python preprocessing/run_all.py --patient brenton --sensor ecg
    python preprocessing/run_all.py --overwrite
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import pandas as pd

from preprocessing.ecg_pipeline import process_patient_ecg
from preprocessing.ppg_pipeline import process_patient_ppg

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "sessions.csv"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess all registered sensor sessions.")
    p.add_argument("--patient", default=None, help="Only process this patient ID")
    p.add_argument("--sensor", default=None, choices=["ecg", "ppg"],
                   help="Only process this sensor type")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess even if output files already exist")
    p.add_argument("--manifest", default=str(MANIFEST_PATH),
                   help="Path to sessions.csv")
    p.add_argument("--processed-dir", default=str(PROCESSED_ROOT),
                   help="Root directory for processed output")
    return p.parse_args()


def run(
    patient: str | None = None,
    sensor: str | None = None,
    overwrite: bool = False,
    manifest: str | Path = MANIFEST_PATH,
    processed_dir: str | Path = PROCESSED_ROOT,
) -> None:
    """
    Run preprocessing for all sessions matching the given filters.

    Groups sessions by (patient, sensor) and calls the appropriate
    process_patient_* function, which handles scaler fitting order
    internally.

    Args:
        patient:       If given, only process this patient
        sensor:        If given, only process this sensor type ("ecg"/"ppg")
        overwrite:     If False, skip groups with all outputs already present
        manifest:      Path to sessions.csv
        processed_dir: Root directory for processed .npy output
    """
    manifest = Path(manifest)
    processed_dir = Path(processed_dir)

    df = pd.read_csv(manifest)

    if patient:
        df = df[df["patient_id"] == patient]
    if sensor:
        df = df[df["sensor_type"] == sensor]

    if df.empty:
        print("No sessions match the given filters.")
        return

    # Group by (patient, sensor) — process_patient_* handles all conditions
    for (pid, stype), group in df.groupby(["patient_id", "sensor_type"]):
        patient_out = processed_dir / pid

        if not overwrite:
            scaler_path = patient_out / f"{stype}_scaler.pkl"
            if scaler_path.exists():
                print(f"Skipping {pid}/{stype} — already processed (use --overwrite to redo)")
                continue

        # Resolve raw file directory from the first row in the group
        first_file = REPO_ROOT / group["file_path"].iloc[0]
        raw_dir = first_file.parent

        print(f"\n=== Processing {pid} / {stype} from {raw_dir} ===")

        if stype == "ecg":
            process_patient_ecg(
                raw_dir=raw_dir,
                patient=pid,
                output_dir=processed_dir,
            )
        elif stype == "ppg":
            process_patient_ppg(
                raw_dir=raw_dir,
                patient=pid,
                output_dir=processed_dir,
            )
        else:
            print(f"Unknown sensor type {stype!r} — skipping")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    args = parse_args()
    run(
        patient=args.patient,
        sensor=args.sensor,
        overwrite=args.overwrite,
        manifest=args.manifest,
        processed_dir=args.processed_dir,
    )
