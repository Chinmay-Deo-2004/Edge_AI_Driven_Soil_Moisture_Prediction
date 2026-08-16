from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def moisture_band(value: float) -> str:
    if value < 35:
        return "dry"
    if value <= 50:
        return "optimum"
    return "saturated"


def _parse_path(path: Path, root: Path) -> tuple[str, float]:
    text = " ".join(path.relative_to(root).with_suffix("").parts).lower()
    soil = next((name for name in ("kaolin", "silt") if name in text), None)
    match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*(?:%|pct|percent)?(?!\d)", text)
    if soil is None or match is None:
        raise ValueError(f"Could not infer soil type and moisture percentage from {path}")
    moisture = float(match.group(1))
    if not 0 <= moisture <= 100:
        raise ValueError(f"Invalid moisture percentage in {path}: {moisture}")
    return soil, moisture


def build_manifest(data_dir: str | Path, labels_csv: str | Path | None = None) -> pd.DataFrame:
    root = Path(data_dir).resolve()
    csv_path = Path(labels_csv) if labels_csv else root.parent / "labels.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        required = {"path", "soil_type", "moisture_pct"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        frame["path"] = frame["path"].map(lambda p: str((root / p).resolve()))
    else:
        rows = []
        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                soil, moisture = _parse_path(path, root)
                rows.append({"path": str(path.resolve()), "soil_type": soil, "moisture_pct": moisture})
        frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No labeled images found under {root}")
    frame["moisture_pct"] = frame["moisture_pct"].astype(float)
    frame["moisture_class"] = frame["moisture_pct"].map(moisture_band)
    return frame.sort_values("path").reset_index(drop=True)