"""Download the frozen dataset ZIP, verify SHA-256, extract read-only.

Refuses to extract or train on a hash mismatch (DATASET_ZIP_HASH_MISMATCH) and
never retries a permission denial from Drive (DOWNLOAD_BLOCKED_BY_DRIVE_PERMISSION).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from common import DATA_ROOT, WORKSPACE, die, find_package_root, log, write_json, OUT

EXPECTED_SHA256 = "1f5b301a7a340ac89620b9124b8df78f82928dbf422d69ffe227f55c1eb4c907"
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "11sk-Ol6p01xT7eC-ktjaI0I4VXnFGedv")
ZIP_PATH = WORKSPACE / "final_dataset_v112.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def download() -> None:
    if ZIP_PATH.is_file() and sha256(ZIP_PATH) == EXPECTED_SHA256:
        log("ZIP already present with the expected hash; skipping download.")
        return

    direct = os.environ.get("DATASET_URL", "").strip()
    if direct:
        log("Downloading from DATASET_URL.")
        rc, out = _run(["curl", "-fSL", "--retry", "4", "--retry-delay", "3",
                        "-o", str(ZIP_PATH), direct])
        if rc != 0:
            die("DATASET_DOWNLOAD_FAILED", f"curl exit {rc}\n{out[-2000:]}")
        return

    log(f"Downloading Drive file {DRIVE_FILE_ID} with gdown.")
    rc, out = _run([sys.executable, "-m", "gdown", "--id", DRIVE_FILE_ID,
                    "-O", str(ZIP_PATH)])
    if rc != 0:
        blocked = any(
            k in out
            for k in ("Permission denied", "Cannot retrieve the public link",
                      "not have permission", "quota", "Access denied", "403")
        )
        if blocked:
            die(
                "DOWNLOAD_BLOCKED_BY_DRIVE_PERMISSION",
                "Drive refused the download. The file must be link-shareable "
                "('Anyone with the link - Viewer') or supplied via DATASET_URL.\n"
                + out[-2000:],
            )
        die("DATASET_DOWNLOAD_FAILED", f"gdown exit {rc}\n{out[-2000:]}")


def verify() -> str:
    if not ZIP_PATH.is_file():
        die("DATASET_DOWNLOAD_FAILED", f"{ZIP_PATH} does not exist after download.")
    actual = sha256(ZIP_PATH)
    log(f"sha256={actual}")
    log(f"expect={EXPECTED_SHA256}")
    if actual != EXPECTED_SHA256:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ".hash_mismatch").write_text(actual, encoding="utf-8")
        die(
            "DATASET_ZIP_HASH_MISMATCH",
            f"Expected {EXPECTED_SHA256}\nActual   {actual}\n"
            "Refusing to extract or train on a dataset that is not the frozen release.",
        )
    log("SHA-256 matches the frozen release.")
    return actual


def extract() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    marker = DATA_ROOT / ".extracted_ok"
    if not marker.exists():
        log(f"Extracting into {DATA_ROOT}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            for member in zf.namelist():
                # Refuse path traversal rather than trusting the archive.
                dest = (DATA_ROOT / member).resolve()
                if not str(dest).startswith(str(DATA_ROOT.resolve())):
                    die("DATASET_ZIP_UNSAFE_PATH", f"Refusing member {member!r}")
            zf.extractall(DATA_ROOT)
        marker.write_text("ok", encoding="utf-8")
    pkg = find_package_root(DATA_ROOT)
    log(f"Package root: {pkg}")
    return pkg


def integrity_manifest(pkg: Path) -> dict:
    """Hash the training inputs so a later stage can prove nothing mutated them.

    The packaged validators may legitimately write inside the tree, so the tree
    is not chmod'ed read-only; instead the inputs that feed the model are
    fingerprinted and re-checked after training.
    """
    entries = {}
    for split in ("train", "validation", "test"):
        f = pkg / "hf_multimodal" / f"{split}.jsonl"
        if f.is_file():
            entries[str(f.relative_to(pkg))] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    img_dir = pkg / "images"
    if img_dir.is_dir():
        imgs = sorted(q for q in img_dir.rglob("*") if q.is_file())
        h = hashlib.sha256()
        for q in imgs:
            h.update(str(q.relative_to(pkg)).encode())
            h.update(sha256(q).encode())
        entries["images/"] = {"file_count": len(imgs), "tree_sha256": h.hexdigest()}
    return entries


def main() -> None:
    download()
    digest = verify()
    pkg = extract()
    counts = {}
    for split in ("train", "validation", "test"):
        p = pkg / "hf_multimodal" / f"{split}.jsonl"
        counts[split] = sum(1 for line in p.open(encoding="utf-8") if line.strip()) if p.is_file() else None
    write_json(
        OUT / "dataset_download.json",
        {
            "zip_sha256": digest,
            "zip_sha256_expected": EXPECTED_SHA256,
            "zip_bytes": ZIP_PATH.stat().st_size,
            "package_root": str(pkg),
            "record_counts": counts,
            "dataset_version": "v1.1.2-independent-oracle-clean",
            "integrity_manifest": integrity_manifest(pkg),
            "DATA_DOWNLOAD_VERIFIED": True,
        },
    )
    log(f"Record counts: {counts}")
    log("DATA_DOWNLOAD_VERIFIED=true")


if __name__ == "__main__":
    main()
