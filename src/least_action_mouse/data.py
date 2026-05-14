from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.request import urlopen


KH2017_RDA_URL = (
    "https://raw.githubusercontent.com/pascalkieslich/mousetrap/master/"
    "data/KH2017_raw.rda"
)


def ensure_kh2017_csv(data_dir: str | Path = "data") -> Path:
    """Download and export KH2017_raw to a local CSV cache."""

    root = Path(data_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rda_path = raw_dir / "KH2017_raw.rda"
    csv_path = processed_dir / "KH2017_raw.csv"
    if csv_path.exists():
        return csv_path

    if not rda_path.exists():
        _download(KH2017_RDA_URL, rda_path)

    _export_rda_to_csv(rda_path, csv_path)
    return csv_path


def _download(url: str, destination: Path) -> None:
    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def _export_rda_to_csv(rda_path: Path, csv_path: Path) -> None:
    rda = rda_path.resolve().as_posix()
    csv = csv_path.resolve().as_posix()
    script = f"""
load("{rda}")
KH2017_raw[] <- lapply(KH2017_raw, function(x) if(is.factor(x)) as.character(x) else x)
write.csv(KH2017_raw, "{csv}", row.names=FALSE, fileEncoding="UTF-8")
"""
    try:
        subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Rscript is required to export KH2017_raw.rda to CSV. "
            "Install R or place KH2017_raw.csv in data/processed/."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Rscript failed while exporting KH2017_raw.rda:\n"
            f"STDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc
