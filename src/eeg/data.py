"""BNCI2014_001 access and the dataset manifest written by `make download`.

Run as `python -m eeg.data`. Fetches every subject into the project-local cache,
hashes the cached files, loads each subject once to count trials, and writes
results/dataset_manifest.json. Strict failure logging ensures nothing is dropped silently.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "dataset_manifest.json"

# MOABB and MNE resolve their cache from MNE_DATA. Keep it project-local unless the
# caller already set it. Must happen before moabb is imported.
os.environ.setdefault("MNE_DATA", str(DATA_DIR))

DATASET_CLASS = "BNCI2014_001"
N_SUBJECTS_EXPECTED = 9
# Must match SPEC.md section 4 (shared preprocessing).
PARADIGM_KWARGS = {"n_classes": 4, "fmin": 8, "fmax": 32}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass(frozen=True)
class SubjectData:
    """One subject, row-aligned: X[i], y[i] and meta.iloc[i] describe the same trial."""

    subject: int
    X: np.ndarray  # (n_trials, n_channels, n_samples), microvolts
    y: np.ndarray  # class names
    meta: pd.DataFrame  # columns subject, session, run (strings; run ids repeat across sessions)
    sfreq: float


def load_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def manifest_subjects(manifest: dict) -> list[int]:
    return sorted(int(s) for s in manifest["subjects"])


def data_available(subjects) -> tuple[bool, str]:
    """True if the manifest exists and every cached file of these subjects is on disk.

    Never touches the network, so tests can skip instead of triggering a download.
    """
    manifest = load_manifest()
    if manifest is None:
        return False, "results/dataset_manifest.json is missing - run `make download`"
    for s in subjects:
        entry = manifest["subjects"].get(str(s))
        if entry is None:
            return False, f"subject {s} is not in the manifest"
        for f in entry["files"]:
            for key in ("path", "nemar_sourcedata_path"):
                rel = f.get(key)
                if rel and not (REPO_ROOT / rel).exists():
                    return False, f"cached file {rel} is missing - run `make download`"
    return True, ""


def load_subject(subject: int) -> SubjectData:
    """Epoch one subject with the paradigm fixed in PARADIGM_KWARGS (SPEC.md section 4)."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(**PARADIGM_KWARGS)
    with warnings.catch_warnings(), mne.utils.use_log_level("ERROR"):
        warnings.filterwarnings("ignore", message="Montage name", category=FutureWarning)
        epochs, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
    X = epochs.get_data() * dataset.unit_factor  # MOABB's own scaling to microvolts
    meta = meta[["subject", "session", "run"]].astype(
        {"subject": int, "session": object, "run": object}
    ).reset_index(drop=True)
    if not (len(X) == len(y) == len(meta)) or set(meta["subject"]) != {subject}:
        raise ValueError(f"subject {subject}: X, y and metadata are not row-aligned")
    return SubjectData(subject, X, np.asarray(y).astype(str), meta, float(epochs.info["sfreq"]))


def _file_urls(subject: int) -> list[str]:
    from moabb.datasets.bnci.base import BNCI_URL

    return [f"{BNCI_URL}001-2014/A{subject:02d}{r}.mat" for r in "TE"]


def _attach_nemar_copy(dataset, rec: dict) -> None:
    """Hash MOABB's NEMAR sourcedata copy of a file and record whether it matches upstream.

    get_data() loads from the NEMAR sourcedata store when the dataset has one, while
    data_path() points at the upstream host. The manifest hashes only mean something for the
    loaded trials if the two copies are byte-identical, so check instead of assuming.
    """
    from moabb.datasets.download import nemar_store

    rec["identical_to_upstream"] = None
    if not dataset.nemar_id:
        return
    copy = nemar_store(dataset.code, dataset.nemar_id) / "sourcedata" / Path(rec["path"]).name
    if not copy.exists():
        return
    rec["nemar_sourcedata_path"] = _rel(copy)
    rec["nemar_sourcedata_sha256"] = sha256_file(copy)
    rec["identical_to_upstream"] = rec["nemar_sourcedata_sha256"] == rec["sha256"]


def _write_failure(moabb_version: str, base_url: str, lines: list[str]) -> None:
    (RESULTS_DIR / "download_failure.txt").write_text(
        f"moabb_version={moabb_version}\nbase_url={base_url}\n" + "\n".join(lines) + "\n"
    )
    print("DOWNLOAD FAILURE - see results/download_failure.txt", file=sys.stderr)


def _describe_subject(dataset, paradigm, subject: int) -> dict:
    """Load one subject and describe it. Raises if anything about the load is off."""
    raws = dataset.get_data([subject])[subject]  # {session: {run: Raw}}
    sfreqs = {raw.info["sfreq"] for runs in raws.values() for raw in runs.values()}
    if len(sfreqs) != 1:
        raise ValueError(f"inconsistent sampling rates across runs: {sorted(sfreqs)}")
    first = next(iter(next(iter(raws.values())).values()))
    eeg_names = [n for n, t in zip(first.ch_names, first.get_channel_types()) if t == "eeg"]

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    if X.shape[0] != len(y) or len(y) != len(meta):
        raise ValueError(f"X/y/metadata length mismatch: {X.shape[0]}, {len(y)}, {len(meta)}")

    per_run = Counter(zip(meta["session"], meta["run"]))
    sessions: dict[str, dict] = {}
    for (session, run), n in sorted(per_run.items()):
        s = sessions.setdefault(str(session), {"n_trials": 0, "runs": {}})
        s["n_trials"] += n
        s["runs"][str(run)] = n

    return {
        "n_trials": int(len(y)),
        "X_shape": [int(d) for d in X.shape],
        "sampling_rate_hz": float(sfreqs.pop()),
        "channel_names": eeg_names,
        "class_counts": {str(k): int(v) for k, v in sorted(Counter(y).items())},
        "sessions": sessions,
    }


def download_and_manifest() -> int:
    import mne
    import moabb
    from moabb.datasets import BNCI2014_001
    from moabb.datasets.bnci.base import BNCI_URL
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(**PARADIGM_KWARGS)
    manifest_path = RESULTS_DIR / "dataset_manifest.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    Path(os.environ["MNE_DATA"]).mkdir(parents=True, exist_ok=True)  # MNE refuses a missing cache dir

    prior = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text())["subjects"].values():
            prior.update({f["path"]: f["sha256"] for f in entry["files"]})

    # Stage 1: fetch + hash. Any failure here is a download failure, not a silent drop.
    files_by_subject: dict[int, list[dict]] = {}
    excluded: list[dict] = []
    failures: list[str] = []
    for s in dataset.subject_list:
        try:
            files = [Path(p) for p in dataset.data_path(s)]
            files_by_subject[s] = [
                {"path": _rel(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in files
            ]
        except Exception as exc:  # noqa: BLE001 - logged and surfaced, never swallowed
            reason = f"{type(exc).__name__}: {exc}"
            excluded.append({"subject": s, "stage": "download", "reason": reason})
            failures.append(f"subject={s}\turls={' '.join(_file_urls(s))}\terror={reason}")

    # Integrity against the previous manifest: a changed hash means the cache is not the
    # data the earlier results were produced from. Do not overwrite the evidence.
    mismatched = [
        f"{f['path']}\tprevious={prior[f['path']]}\tnow={f['sha256']}"
        for files in files_by_subject.values()
        for f in files
        if f["path"] in prior and prior[f["path"]] != f["sha256"]
    ]
    if mismatched:
        failures.append("checksum mismatch vs existing manifest:\n  " + "\n  ".join(mismatched))

    if failures:
        _write_failure(moabb.__version__, BNCI_URL, failures)
        return 1

    # Stage 2: load each subject once and describe it.
    subjects: dict[str, dict] = {}
    for s, files in files_by_subject.items():
        try:
            entry = _describe_subject(dataset, paradigm, s)
        except Exception as exc:  # noqa: BLE001 - logged and surfaced, never swallowed
            excluded.append({"subject": s, "stage": "load", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        for f in files:
            _attach_nemar_copy(dataset, f)  # after the load, which is what fetches the NEMAR copy
        entry["files"] = files
        subjects[str(s)] = entry

    diverged = [
        f"{f['path']}\tupstream={f['sha256']}\tnemar={f['nemar_sourcedata_sha256']}"
        for e in subjects.values()
        for f in e["files"]
        if f["identical_to_upstream"] is False
    ]
    if diverged:
        _write_failure(
            moabb.__version__,
            BNCI_URL,
            ["NEMAR sourcedata differs from upstream file (loaded data != hashed data):\n  "
             + "\n  ".join(diverged)],
        )
        return 1
    (RESULTS_DIR / "download_failure.txt").unlink(missing_ok=True)

    (RESULTS_DIR / "excluded_subjects.txt").write_text(
        "".join(f"{e['subject']}\t{e['stage']}\t{e['reason']}\n" for e in excluded)
    )

    classes = sorted({c for e in subjects.values() for c in e["class_counts"]})
    sfreqs = sorted({e["sampling_rate_hz"] for e in subjects.values()})
    totals = Counter()
    for e in subjects.values():
        totals.update(e["class_counts"])
    manifest = {
        "dataset": {
            "class": DATASET_CLASS,
            "code": dataset.code,
            "doi": getattr(dataset, "doi", None),
            "base_url": BNCI_URL,
            "nemar_id": dataset.nemar_id,
            "interval_s": list(dataset.interval),
        },
        "versions": {
            "moabb": moabb.__version__,
            "mne": mne.__version__,
            "python": sys.version.split()[0],
        },
        "download_date_utc": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "cache_dir": _rel(os.environ["MNE_DATA"]),
        "paradigm": {
            "name": type(paradigm).__name__,
            "n_classes": paradigm.n_classes,
            "filters": paradigm.filters,
            "tmin": paradigm.tmin,
            "tmax": paradigm.tmax,
            "resample": paradigm.resample,
        },
        "n_subjects_expected": N_SUBJECTS_EXPECTED,
        "n_subjects_used": len(subjects),
        "n_classes": len(classes),
        "classes": classes,
        "chance_level": 1 / len(classes) if classes else None,
        "max_class_fraction": max(totals.values()) / sum(totals.values()) if totals else None,
        "sampling_rate_hz": sfreqs[0] if len(sfreqs) == 1 else sfreqs,
        "subjects": subjects,
        "excluded_subjects": excluded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{'subject':>7} {'n_trials':>8}  {'X_shape':<18} class_counts / sessions")
    for s, e in subjects.items():
        sess = {k: v["n_trials"] for k, v in e["sessions"].items()}
        print(f"{s:>7} {e['n_trials']:>8}  {str(tuple(e['X_shape'])):<18} {e['class_counts']} {sess}")
    for e in excluded:
        print(f"EXCLUDED subject {e['subject']} ({e['stage']}): {e['reason']}", file=sys.stderr)
    print(f"n_subjects_used={len(subjects)}/{N_SUBJECTS_EXPECTED}  manifest: {_rel(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(download_and_manifest())
