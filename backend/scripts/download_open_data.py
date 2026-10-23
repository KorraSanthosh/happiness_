#!/usr/bin/env python3
"""
Download the openly licensed (CC BY 4.0) datasets used to train GlucoTwin v7 into <repo>/external_data/.

  cd backend && python3 scripts/download_open_data.py

Datasets (please cite them - see the README):
  hupa_ucm  HUPA-UCM Diabetes Dataset, Mendeley Data 10.17632/3hbcscwz44.1 (preprocessed CSVs, ~27 MB)
  t1d_uom   T1D-UOM, Zenodo 10.5281/zenodo.15806142 (~7 MB)
  azt1d     AZT1D, Mendeley Data 10.17632/gk9m674wcx.1 (~776 MB zip)
  shanghai  ShanghaiT1DM/T2DM, figshare 10.6084/m9.figshare.20444397 (~3 MB)

Not downloaded here:
  BrisT1D   Kaggle "BrisT1D Blood Glucose Prediction" (needs a Kaggle account) -> put train.csv in <repo>/brist1d_raw/
  OhioT1DM  data-use agreement with Ohio University; optional external test -> <repo>/ohiot1dm_raw/

Downloaded archives are extracted with path-traversal checks; nothing inside them is ever executed.
"""
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external_data"
UA = {"User-Agent": "Mozilla/5.0 (GlucoTwin data downloader)"}

ZENODO_UOM = "https://zenodo.org/api/records/15806142/files/sharpic/ManchesterCSCoordinatedDiabetesStudy-V1.0.3.zip/content"
FIGSHARE_SHANGHAI = "https://ndownloader.figshare.com/files/38259264"
MENDELEY = "https://data.mendeley.com/public-api/datasets/{ds}"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"  downloaded: {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = (dest / name).resolve()
            if target != dest and not str(target).startswith(str(dest) + os.sep):
                raise SystemExit(f"unsafe path in {zip_path.name}: {name}")
        z.extractall(dest)
    print(f"  extracted: {zip_path.name} -> {dest.relative_to(ROOT)}")


def mendeley_files(ds: str):
    get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers={**UA, "Accept": "application/json"})))
    folders = get(f"{MENDELEY.format(ds=ds)}/folders/1")
    byid = {f["id"]: f for f in folders}

    def path(fid):
        parts = []
        while fid in byid:
            parts.append(byid[fid]["name"])
            fid = byid[fid].get("parent_id")
        return "/".join(reversed(parts))

    seen = set()
    for fid in list(byid) + ["root"]:
        try:
            files = get(f"{MENDELEY.format(ds=ds)}/files?folder_id={fid}&version=1")
        except Exception:
            continue
        for f in files:
            url = f["content_details"]["download_url"]
            if url not in seen:
                seen.add(url)
                yield path(fid), f["filename"], url


def main() -> None:
    EXT.mkdir(exist_ok=True)
    print("HUPA-UCM (preprocessed CSVs)")
    for folder, name, url in mendeley_files("3hbcscwz44"):
        if folder == "Preprocessed" and name.endswith(".csv") and "/" not in name:
            fetch(url, EXT / "hupa_ucm" / name)
    print("T1D-UOM")
    fetch(ZENODO_UOM, EXT / "t1d_uom" / "t1d_uom.zip")
    safe_extract(EXT / "t1d_uom" / "t1d_uom.zip", EXT / "t1d_uom" / "extracted")
    print("Shanghai T1DM/T2DM")
    fetch(FIGSHARE_SHANGHAI, EXT / "shanghai" / "data.zip")
    safe_extract(EXT / "shanghai" / "data.zip", EXT / "shanghai" / "extracted")
    print("AZT1D (large, ~776 MB)")
    for _, name, url in mendeley_files("gk9m674wcx"):
        if name.endswith(".zip"):
            fetch(url, EXT / "azt1d" / "AZT1D_2025.zip")
    safe_extract(EXT / "azt1d" / "AZT1D_2025.zip", EXT / "azt1d" / "extracted")
    print("\nDone. Next: python -m app.real_data_v7 --stage all --rebuild")


if __name__ == "__main__":
    sys.exit(main())
