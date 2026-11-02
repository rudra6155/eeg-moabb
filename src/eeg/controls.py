"""Label-shuffle negative control protocol.

The FULL pipeline is run end-to-end with labels randomly permuted within each subject.
If the audit and the pipelines are sound the accuracy sits on chance. Anything materially
above chance means leakage or a preprocessing bug, and the run stops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eeg import cv
from eeg.data import REPO_ROOT, RESULTS_DIR, load_manifest, load_subject, manifest_subjects
from eeg.evaluate import PROTOCOLS, SEED, SMOKE_SUBJECTS, default_configs, evaluate

CONTROL_COLUMNS = ["control", "pipeline", "subject", "accuracy", "n_trials", "chance_level"]
CONTROL_NAMES = {cv.RUN_GROUPED: "label_shuffle", cv.CROSS_SESSION: "label_shuffle_cross_session"}
# Full run: the mean over 9 subjects must sit within 0.03 of chance (~3.5 sigma of the
# sampling noise at 576 trials/subject). Smoke run: only 2 subjects, where 0.03 would be
# ~2 sigma and trip on noise, so the empirical 0.10 sanity threshold applies instead.
AGGREGATE_TOLERANCE = 0.03
SMOKE_AGGREGATE_TOLERANCE = 0.10
SUBJECT_TOLERANCE = 0.10  # no single subject may sit this far above chance


def shuffle_labels(y, session, subject: int, seed: int = SEED) -> np.ndarray:
    """Permute one subject's labels among the trials of each session.

    Deterministic in (seed, subject). Permuting inside a session, not across the whole
    subject, keeps every session's class counts exactly balanced, so the cross-session test
    set (session 1test alone) still has chance level exactly 1/n_classes.
    """
    y, session = np.asarray(y), np.asarray(session)
    out, rng = y.copy(), np.random.default_rng([seed, subject])
    for s in sorted(set(session)):
        idx = np.flatnonzero(session == s)
        out[idx] = y[rng.permutation(idx)]
    return out


def run_controls(subjects, loader=load_subject, configs=None, protocols=PROTOCOLS, seed: int = SEED):
    configs = configs or default_configs()
    rows = []
    for subject in subjects:
        data = loader(subject)
        y_shuffled = shuffle_labels(data.y, data.meta["session"], subject, seed)
        for protocol in protocols:
            splits = cv.make_splits(protocol, data.meta)
            for name, config in configs.items():
                score = evaluate(config, data.X, y_shuffled, data.meta, splits, seed=seed)
                rows.append(
                    {
                        "control": CONTROL_NAMES[protocol],
                        "pipeline": name,
                        "subject": subject,
                        "accuracy": score.accuracy,
                        "n_trials": score.n_trials,
                        "chance_level": score.chance_level,
                    }
                )
                print(f"[{CONTROL_NAMES[protocol]}] subject {subject} {name}: acc={score.accuracy:.3f}", flush=True)
    return pd.DataFrame(rows, columns=CONTROL_COLUMNS)


def judge(df: pd.DataFrame, aggregate_tolerance: float = AGGREGATE_TOLERANCE) -> list[str]:
    """Reasons the control failed (empty list = at chance)."""
    problems = []
    for (control, pipeline), part in df.groupby(["control", "pipeline"]):
        chance = part["chance_level"].iloc[0]
        mean = part["accuracy"].mean()
        if abs(mean - chance) > aggregate_tolerance:
            problems.append(f"{control}/{pipeline}: mean accuracy {mean:.3f} vs chance {chance:.3f} "
                            f"(tolerance {aggregate_tolerance})")
        for _, r in part[part["accuracy"] > chance + SUBJECT_TOLERANCE].iterrows():
            problems.append(f"{control}/{pipeline}: subject {int(r['subject'])} at {r['accuracy']:.3f}, "
                            f"more than {SUBJECT_TOLERANCE} above chance {chance:.3f}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Label-shuffle negative control.")
    parser.add_argument("--smoke", action="store_true", help="2 subjects into results_smoke/")
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.smoke:
        subjects, out_dir = args.subjects or SMOKE_SUBJECTS, args.out or REPO_ROOT / "results_smoke"
    else:
        manifest = load_manifest()
        if manifest is None:
            print("results/dataset_manifest.json is missing - run `make download` first", file=sys.stderr)
            return 1
        subjects, out_dir = args.subjects or manifest_subjects(manifest), args.out or RESULTS_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_controls(subjects)
    df.to_csv(out_dir / "controls.csv", index=False)
    summary = df.groupby(["control", "pipeline"])["accuracy"].agg(["mean", "std", "count"])
    print("\n" + summary.to_string())

    problems = judge(df, SMOKE_AGGREGATE_TOLERANCE if args.smoke else AGGREGATE_TOLERANCE)
    if problems:
        note = RESULTS_DIR / "leakage_investigation.txt"
        RESULTS_DIR.mkdir(exist_ok=True)
        note.write_text(
            "Label-shuffle control is above chance. This is a bug to investigate, not a result.\n"
            "Suspected stages to check: scaling, CSP / reference-covariance fit, split construction.\n\n"
            + "\n".join(problems) + f"\n\nsubjects={list(subjects)}  output={out_dir}\n"
        )
        print("CONTROL FAILED - see results/leakage_investigation.txt", file=sys.stderr)
        return 1
    print("\nlabel-shuffle control is at chance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
