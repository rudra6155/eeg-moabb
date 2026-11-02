"""Evaluation engine, aggregation, and the `python -m eeg.evaluate` entry point (run-all / smoke).

`evaluate()` takes a frozen pipeline config, never a search space. `run_splits()` is the
engine underneath: it audits every split and every fit (leakage.py) before a score exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import platform
import random
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection._search import BaseSearchCV

from eeg import csp_lda, cv, riemann
from eeg.csp_lda import CspLdaConfig
from eeg.data import REPO_ROOT, RESULTS_DIR, load_manifest, load_subject, manifest_subjects
from eeg.leakage import FitWatcher, LeakageError, audit_splits
from eeg.riemann import RiemannConfig

SEED = 42
HEADLINE = cv.RUN_GROUPED
SECONDARY = cv.CROSS_SESSION
PROTOCOLS = (HEADLINE, SECONDARY)
SMOKE_SUBJECTS = [1, 2]

SCORE_COLUMNS = ["subject", "pipeline", "accuracy", "balanced_accuracy", "n_trials", "chance_level"]
AGG_COLUMNS = [
    "pipeline",
    "mean_accuracy",
    "std_accuracy",
    "mean_kappa",
    "std_kappa",
    "n_subjects",
    "chance_level",
    "p_value_vs_chance",
    "test_used",
]
TEST_USED = "exact one-sided sign-flip permutation test on subject-level accuracy minus chance"

BUILDERS = {CspLdaConfig: csp_lda.build, RiemannConfig: riemann.build}
_SINGLE_VALUE = (int, float, str, bool, type(None))


class SchemaError(ValueError):
    """A per-subject score table is missing rows or is malformed. Always a hard failure."""


def default_configs() -> dict:
    return {cls.name: cls() for cls in BUILDERS}


def seed_all(seed: int) -> None:
    """Seed python, numpy (which scikit-learn and pyriemann draw from)."""
    random.seed(seed)
    np.random.seed(seed)


# ---- configs: fixed values only --------------------------------------------------------


def validate_config(config) -> None:
    """Reject anything that is not one frozen config holding single values.

    A search space (dict, list, GridSearchCV, ...) would let hyperparameters be chosen by
    looking at scores, which on test folds is exactly the tuning the benchmark forbids.
    """
    if isinstance(config, type) or not dataclasses.is_dataclass(config):
        raise TypeError(f"evaluate() takes a frozen pipeline config, got {type(config).__name__}")
    if not config.__dataclass_params__.frozen:
        raise TypeError(f"{type(config).__name__} must be a frozen dataclass")
    if type(config) not in BUILDERS:
        raise TypeError(f"no pipeline builder registered for {type(config).__name__}")
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if not isinstance(value, _SINGLE_VALUE):
            raise TypeError(f"{f.name}={value!r} is not a single fixed value; search spaces are not allowed")


def build_pipeline(config):
    validate_config(config)
    return BUILDERS[type(config)](config)


def _reject_search(estimator) -> None:
    stack = [estimator]
    while stack:
        e = stack.pop()
        if isinstance(e, BaseSearchCV) or type(e).__name__.endswith("CV"):
            raise TypeError(f"{type(e).__name__} selects hyperparameters by cross-validation; not allowed")
        stack.extend(step for _, step in getattr(e, "steps", []))


# ---- scoring ---------------------------------------------------------------------------


def chance_level(y) -> float:
    """1 / n_classes, asserted against the actual class balance of the scored trials."""
    _, counts = np.unique(y, return_counts=True)
    chance = 1.0 / len(counts)
    if not np.allclose(counts / counts.sum(), chance):
        raise ValueError(f"classes are not balanced ({counts.tolist()}); 1/n_classes is not the chance level")
    return chance


@dataclass(frozen=True)
class SubjectScore:
    subject: int
    pipeline: str
    accuracy: float
    balanced_accuracy: float
    n_trials: int
    chance_level: float
    y_true: np.ndarray
    y_pred: np.ndarray

    def row(self) -> dict:
        return {c: getattr(self, c) for c in SCORE_COLUMNS}


def run_splits(build, X, y, meta, splits):
    """Fit on each training fold, predict each test fold, audit everything on the way.

    `build` is a zero-argument callable returning a fresh unfitted estimator. Returns the
    out-of-fold (y_true, y_pred), concatenated in split order.
    """
    import mne

    audit_splits(splits, meta)
    y_true, y_pred = [], []
    with FitWatcher() as watcher, mne.utils.use_log_level("ERROR"):
        for split in splits:
            watcher.reset()
            estimator = build()
            _reject_search(estimator)
            estimator.fit(X[split.train_idx], y[split.train_idx])
            pred = estimator.predict(X[split.test_idx])
            watcher.check(split, X)
            y_true.append(y[split.test_idx])
            y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate(config, X, y, meta, splits, *, seed: int = SEED) -> SubjectScore:
    validate_config(config)
    seed_all(seed)
    y_true, y_pred = run_splits(partial(build_pipeline, config), X, y, meta, splits)
    return SubjectScore(
        subject=splits[0].subject,
        pipeline=config.name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        n_trials=int(len(y_true)),
        chance_level=chance_level(y_true),
        y_true=y_true,
        y_pred=y_pred,
    )


# ---- aggregation -----------------------------------------------------------------------


def sign_flip_pvalue(d) -> float:
    """Exact one-sided sign-flip permutation p-value for H1: mean(d) > 0.

    Under H0 each subject's deviation from chance is symmetric around 0, so every sign
    pattern is equally likely. p = share of the 2^n patterns whose sum is >= the observed.
    """
    d = np.asarray(d, dtype=float)
    if not 1 <= len(d) <= 20:
        raise ValueError("the exact test enumerates 2^n sign patterns; need 1 <= n <= 20 subjects")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(signs @ d >= d.sum() - 1e-12))


def kappa_from_accuracy(accuracy, chance):
    """Cohen's kappa from accuracy: (accuracy - chance) / (1 - chance).

    Exact, not an approximation, when the true classes are balanced: the expected
    agreement is sum_k p_true(k) * p_pred(k) = (1/K) * sum_k p_pred(k) = 1/K whatever the
    predictions are. tests/test_aggregate.py checks it against sklearn's cohen_kappa_score.
    """
    return (np.asarray(accuracy, dtype=float) - chance) / (1.0 - chance)


def validate_scores(df: pd.DataFrame, expected_subjects, expected_pipelines) -> None:
    if list(df.columns) != SCORE_COLUMNS:
        raise SchemaError(f"columns must be exactly {SCORE_COLUMNS}, got {list(df.columns)}")
    if df.isna().any().any():
        raise SchemaError("score table contains empty cells")
    if df.duplicated(["subject", "pipeline"]).any():
        raise SchemaError("duplicate (subject, pipeline) rows")
    expected = {(int(s), p) for s in expected_subjects for p in expected_pipelines}
    found = set(zip(df["subject"].astype(int), df["pipeline"]))
    if expected - found:
        raise SchemaError(f"missing (subject, pipeline) rows: {sorted(expected - found)}")
    if found - expected:
        raise SchemaError(f"unexpected (subject, pipeline) rows: {sorted(found - expected)}")
    for col in ("accuracy", "balanced_accuracy"):
        if not df[col].between(0, 1).all():
            raise SchemaError(f"{col} must lie in [0, 1]")
    if not (df["n_trials"] > 0).all() or not df["chance_level"].between(0, 1, inclusive="neither").all():
        raise SchemaError("n_trials must be positive and chance_level must lie in (0, 1)")
    if (df.groupby("pipeline")["chance_level"].nunique() != 1).any():
        raise SchemaError("chance_level differs between subjects of one pipeline")


def aggregate(scores_csv, out_csv, expected_subjects, expected_pipelines) -> pd.DataFrame:
    """Per-pipeline mean/std (ddof=1) and the sign-flip p-value. Fails on any missing or invalid row."""
    df = pd.read_csv(scores_csv)
    validate_scores(df, expected_subjects, expected_pipelines)
    rows = []
    for pipeline in expected_pipelines:
        part = df[df["pipeline"] == pipeline]
        acc, chance = part["accuracy"].to_numpy(), float(part["chance_level"].iloc[0])
        kappa = kappa_from_accuracy(acc, chance)
        rows.append(
            {
                "pipeline": pipeline,
                "mean_accuracy": float(acc.mean()),
                "std_accuracy": float(acc.std(ddof=1)) if len(acc) > 1 else float("nan"),
                "mean_kappa": float(kappa.mean()),
                "std_kappa": float(kappa.std(ddof=1)) if len(acc) > 1 else float("nan"),
                "n_subjects": int(len(acc)),
                "chance_level": chance,
                "p_value_vs_chance": sign_flip_pvalue(acc - chance),
                "test_used": TEST_USED,
            }
        )
    out = pd.DataFrame(rows, columns=AGG_COLUMNS)
    out.to_csv(out_csv, index=False)
    return out


# ---- figures ---------------------------------------------------------------------------

PIPELINE_LABELS = {"csp_lda": "CSP + LDA", "riemann_ts_lr": "Riemannian tangent space + LR"}

# Light-mode tokens and categorical slots 1-2 from the dataviz reference palette
# (validate_palette.js: all checks pass for these two) and its one-hue blue sequential ramp.
_INK = {
    "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
    "grid": "#e1e0d9", "baseline": "#c3c2b7",
}
_SERIES = ["#2a78d6", "#eb6834"]
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_DPI = 150
_PX = _DPI / 96  # device pixels per CSS pixel, so the mark specs (24px, 4px, 2px) hold in the PNG


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": _INK["primary"],
    })
    return plt


def _rounded_bar(ax, x0, w, h, color, r_x, aspect):
    """Bar from the baseline up: 4px rounded data end, square at the baseline."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    r_y = r_x * aspect
    if h <= 2 * r_y:
        ax.add_patch(Rectangle((x0, 0), w, h, fc=color, ec="none"))
        return
    ax.add_patch(FancyBboxPatch((x0, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r_x}",
                                mutation_aspect=aspect, fc=color, ec="none"))
    ax.add_patch(Rectangle((x0, 0), w, r_y, fc=color, ec="none"))  # squares off the baseline end


def _plot_accuracy(out_dir: Path, scores: pd.DataFrame, table: pd.DataFrame) -> None:
    from matplotlib.patches import Patch

    plt = _pyplot()
    W, H = 9.0, 4.8
    left, bottom, width, height = 0.08, 0.14, 0.80, 0.62
    fig = plt.figure(figsize=(W, H), dpi=_DPI, facecolor=_INK["surface"])
    ax = fig.add_axes([left, bottom, width, height], facecolor=_INK["surface"])

    subjects = sorted(scores["subject"].unique())
    pipelines = list(table["pipeline"])
    chance = float(table["chance_level"].iloc[0])
    ax.set_xlim(-0.5, len(subjects) - 0.5)
    ax.set_ylim(0, 1)
    ux = len(subjects) / (width * W * _DPI)  # x data units per device pixel
    uy = 1.0 / (height * H * _DPI)
    bar_w, gap = 24 * _PX, 2 * _PX
    group_w = len(pipelines) * bar_w + (len(pipelines) - 1) * gap
    for j, pipeline in enumerate(pipelines):
        acc = scores[scores["pipeline"] == pipeline].set_index("subject")["accuracy"]
        for i, subject in enumerate(subjects):
            x0 = i + (-group_w / 2 + j * (bar_w + gap)) * ux
            _rounded_bar(ax, x0, bar_w * ux, float(acc[subject]), _SERIES[j], 4 * _PX * ux, uy / ux)

    ax.axhline(chance, color=_INK["secondary"], lw=1.2, zorder=3)
    ax.text(1.01, chance, f"chance {chance:.2f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.5, color=_INK["secondary"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(range(len(subjects)), [str(s) for s in subjects])
    ax.yaxis.grid(True, color=_INK["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, labelcolor=_INK["secondary"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK["baseline"])
    ax.set_xlabel("Subject", color=_INK["secondary"], fontsize=9)
    ax.set_ylabel("Accuracy", color=_INK["secondary"], fontsize=9)

    labels = []
    for pipeline in pipelines:
        row = table[table["pipeline"] == pipeline].iloc[0]
        labels.append(f"{PIPELINE_LABELS.get(pipeline, pipeline)}: mean {row['mean_accuracy']:.3f}, "
                      f"std {row['std_accuracy']:.3f}")
    fig.legend([Patch(fc=c) for c in _SERIES[: len(pipelines)]], labels, loc="upper left",
               bbox_to_anchor=(left - 0.005, 0.885), ncol=1, frameon=False, fontsize=9,
               labelcolor=_INK["secondary"], handlelength=1, handleheight=1)
    fig.text(left, 0.945, "Accuracy per subject", fontsize=13, fontweight="semibold")
    fig.text(left, 0.91, f"4-class motor imagery, run-grouped CV (leave one run out within each session), "
             f"{len(subjects)} subjects", fontsize=9, color=_INK["secondary"])
    fig.savefig(out_dir / "accuracy_per_subject.png", facecolor=_INK["surface"])
    plt.close(fig)


def _text_on(rgb) -> str:
    """White or near-black, whichever contrasts more with the cell colour."""
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return "#ffffff" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / 0.0555) else "#0b0b0b"


def _plot_confusion(out_dir: Path, confusion: dict, labels) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    plt = _pyplot()
    cmap = LinearSegmentedColormap.from_list("blue_sequential", _BLUE_RAMP)
    names = list(confusion)
    fig, axes = plt.subplots(1, len(names), figsize=(4.7 * len(names), 5.3), dpi=_DPI,
                             facecolor=_INK["surface"], squeeze=False)
    n = len(labels)
    classes = [str(l).replace("_", " ") for l in labels]
    for ax, name in zip(axes[0], names):
        counts = confusion[name]
        rate = counts / counts.sum(axis=1, keepdims=True)
        ax.set_facecolor(_INK["surface"])
        ax.pcolormesh(rate, cmap=cmap, vmin=0, vmax=1, edgecolors=_INK["surface"], linewidth=2 * _PX * 72 / _DPI)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)  # first class on top, diagonal runs top-left to bottom-right
        ax.set_aspect("equal")
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.5, i + 0.5, f"{rate[i, j] * 100:.0f}%", ha="center", va="center",
                        fontsize=10, color=_text_on(cmap(rate[i, j])))
        ax.set_xticks(np.arange(n) + 0.5, classes)
        ax.set_yticks(np.arange(n) + 0.5, classes)
        ax.tick_params(length=0, labelsize=8.5, labelcolor=_INK["secondary"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted", color=_INK["secondary"], fontsize=9)
        ax.set_ylabel("True", color=_INK["secondary"], fontsize=9)
        ax.set_title(f"{PIPELINE_LABELS.get(name, name)}  ({int(counts.sum())} trials)", loc="left",
                     fontsize=10.5, fontweight="semibold", color=_INK["primary"], pad=10)
    fig.suptitle("Out-of-fold confusion, all subjects pooled", x=0.02, ha="left", fontsize=13,
                 fontweight="semibold")
    fig.text(0.02, 0.905, "Run-grouped CV. Each row sums to 100% of that true class.",
             fontsize=9, color=_INK["secondary"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "confusion.png", facecolor=_INK["surface"])
    plt.close(fig)


def write_plots(out_dir: Path, scores: pd.DataFrame, table: pd.DataFrame, confusion: dict, labels) -> None:
    _plot_accuracy(Path(out_dir), scores, table)
    _plot_confusion(Path(out_dir), confusion, labels)


# ---- run-all / smoke -------------------------------------------------------------------


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def write_hardware(out_dir: Path, configs: dict) -> None:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": SEED,
        "versions": {p: _version(p) for p in ("moabb", "mne", "pyriemann", "scikit-learn", "numpy", "scipy", "pandas")},
        "configs": {name: dataclasses.asdict(cfg) for name, cfg in configs.items()},
    }
    (out_dir / "hardware.json").write_text(json.dumps(info, indent=2) + "\n")


def _append_unique(path: Path, lines: list[str]) -> None:
    existing = path.read_text().splitlines() if path.exists() else []
    path.write_text("".join(l + "\n" for l in existing + [x for x in lines if x not in existing]))


def run_all(subjects, out_dir, loader=load_subject, configs=None) -> dict:
    """Both pipelines x both protocols for every subject. Returns the aggregate tables.

    Headline files go to `out_dir`, secondary ones to `out_dir/cross_session`. A subject or
    pipeline that fails is logged and stays in the denominator, so aggregation then fails
    loudly instead of silently averaging over fewer subjects.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = configs or default_configs()
    scores = {p: [] for p in PROTOCOLS}
    confusion, labels = {}, None
    excluded, failures = [], []

    for subject in subjects:
        try:
            data = loader(subject)
        except Exception as exc:  # noqa: BLE001 - logged, and aggregation will refuse to continue
            excluded.append(f"{subject}\trun\t{type(exc).__name__}: {exc}")
            continue
        for protocol in PROTOCOLS:
            splits = cv.make_splits(protocol, data.meta)
            for name, config in configs.items():
                try:
                    score = evaluate(config, data.X, data.y, data.meta, splits)
                except LeakageError:
                    raise  # a leak is a bug to investigate, never a per-subject failure
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        json.dumps({"subject": subject, "pipeline": name, "protocol": protocol,
                                    "exception": f"{type(exc).__name__}: {exc}"})
                    )
                    traceback.print_exc()
                    continue
                scores[protocol].append(score.row())
                print(f"[{protocol}] subject {subject} {name}: acc={score.accuracy:.3f} "
                      f"bal={score.balanced_accuracy:.3f} n={score.n_trials}", flush=True)
                if protocol == HEADLINE:
                    labels = sorted(set(score.y_true))
                    cm = confusion_matrix(score.y_true, score.y_pred, labels=labels)
                    confusion[name] = confusion.get(name, 0) + cm

    _append_unique(out_dir / "excluded_subjects.txt", excluded)
    failure_file = out_dir / "pipeline_failures.txt"
    if failures:
        _append_unique(failure_file, failures)
    else:
        failure_file.unlink(missing_ok=True)

    write_hardware(out_dir, configs)
    tables = {}
    for protocol in PROTOCOLS:
        target = out_dir if protocol == HEADLINE else out_dir / "cross_session"
        target.mkdir(exist_ok=True)
        csv = target / "per_subject_scores.csv"
        pd.DataFrame(scores[protocol], columns=SCORE_COLUMNS).to_csv(csv, index=False)
        tables[protocol] = aggregate(csv, target / "aggregate.csv", subjects, list(configs))
    write_plots(out_dir, pd.DataFrame(scores[HEADLINE], columns=SCORE_COLUMNS), tables[HEADLINE], confusion, labels)
    return {"aggregate": tables, "confusion": confusion, "labels": labels}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run both pipelines on both protocols.")
    parser.add_argument("--smoke", action="store_true", help="2 subjects into results_smoke/")
    parser.add_argument("--subjects", type=int, nargs="*", help="override the subject list")
    parser.add_argument("--out", type=Path, help="override the output directory")
    args = parser.parse_args(argv)

    if args.smoke:
        subjects, out_dir = args.subjects or SMOKE_SUBJECTS, args.out or REPO_ROOT / "results_smoke"
    else:
        manifest = load_manifest()
        if manifest is None:
            print("results/dataset_manifest.json is missing - run `make download` first", file=sys.stderr)
            return 1
        subjects, out_dir = args.subjects or manifest_subjects(manifest), args.out or RESULTS_DIR

    try:
        result = run_all(subjects, out_dir)
    except SchemaError as exc:
        print(f"AGGREGATION FAILED: {exc}", file=sys.stderr)
        print(f"see {out_dir}/pipeline_failures.txt and {out_dir}/excluded_subjects.txt", file=sys.stderr)
        return 1

    for protocol, table in result["aggregate"].items():
        print(f"\n== {protocol} ==")
        print(table.drop(columns="test_used").to_string(index=False))
    if args.smoke:
        weak = result["aggregate"][HEADLINE].query("mean_accuracy <= chance_level")["pipeline"].tolist()
        if weak:
            print(f"SMOKE FAILED: not above chance on the headline protocol: {weak}", file=sys.stderr)
            return 1
        print("\nsmoke: both pipelines are above chance on the headline protocol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
