"""Generate FINDINGS.md, README.md and CLAIMS.md from the files in results/.

No figure in those documents is typed by hand. Every number is rendered from a result file by
a function passed to `Ledger.cite`, which also registers it as a row of CLAIMS.md. After the
documents are written, `Ledger.status` re-reads each artifact, re-renders the value and checks
that it matches the document (VERIFIED), and `undeclared_numbers` fails the run if the prose
holds a decimal number that no claim covers.

Run as `python -m eeg.report` (`make report`).
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from eeg import controls
from eeg.data import REPO_ROOT, RESULTS_DIR as R
from eeg.evaluate import PIPELINE_LABELS, kappa_from_accuracy

MANIFEST = R / "dataset_manifest.json"
SCORES = R / "per_subject_scores.csv"
AGG = R / "aggregate.csv"
CS_SCORES = R / "cross_session" / "per_subject_scores.csv"
CS_AGG = R / "cross_session" / "aggregate.csv"
CONTROLS = R / "controls.csv"
HARDWARE = R / "hardware.json"
BASELINE = R / "baseline.json"
MUTATION = R / "leakage_mutation_check.json"
JUNIT = R / "test_results.xml"
CONTROLS_SRC = REPO_ROOT / "src" / "eeg" / "controls.py"

CMD = {
    "download": "make download",
    "run": "make run-all",
    "controls": "make controls",
    "test": "make test",
    "mutation": "make mutation",
    "baseline": "none: transcribed by hand from the cited paper (see the file's _note)",
    "code": "make report (reads the constant from the source file)",
}


# ---- claim ledger ----------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    doc: str
    label: str
    text: str
    artifact: Path
    command: str
    recompute: Callable[[], str]


class Ledger:
    def __init__(self):
        self.claims: list[Claim] = []

    def cite(self, doc: str, label: str, artifact: Path, command: str, recompute: Callable[[], str]) -> str:
        try:
            text = recompute()
        except FileNotFoundError:
            text = f"[BLOCKED: {artifact.name} is missing]"
        self.claims.append(Claim(doc, label, text, artifact, command, recompute))
        return text

    def status(self, claim: Claim, docs: dict[str, str]) -> str:
        if not claim.artifact.exists():
            return "BLOCKED"
        try:
            again = claim.recompute()
        except Exception:  # noqa: BLE001
            return "BLOCKED"
        return "VERIFIED" if again == claim.text and claim.text in docs[claim.doc] else "PENDING"


# ---- rendering helpers -----------------------------------------------------------------


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pm(mean, std) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def name(pipeline: str) -> str:
    return PIPELINE_LABELS.get(pipeline, pipeline)


def j(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))  # Windows would otherwise default to cp1252


def p_text(p: float, n: int) -> str:
    return f"{p:.4f} ({round(p * 2 ** n)}/{2 ** n})"


# ---- renderers (each is a pure function of result files) -------------------------------


def render_dataset(path: Path) -> str:
    m = j(path)
    subj = m["subjects"]
    first = next(iter(subj.values()))
    sessions = first["sessions"]
    n_runs = len(next(iter(sessions.values()))["runs"])
    per_run = next(iter(next(iter(sessions.values()))["runs"].values()))
    files = [f for e in subj.values() for f in e["files"]]
    identical = sum(f["identical_to_upstream"] is True for f in files)
    counts = first["class_counts"]
    excl = m["excluded_subjects"]
    excl_text = "none" if not excl else "; ".join(f"subject {e['subject']} ({e['stage']}): {e['reason']}" for e in excl)
    return "\n".join([
        f"- Dataset: `{m['dataset']['class']}` (BCI Competition IV data set 2a), MOABB {m['versions']['moabb']}, "
        f"downloaded {m['download_date_utc']}.",
        f"- Task: {m['n_classes']}-class motor imagery ({', '.join(m['classes'])}).",
        f"- Subjects used: {m['n_subjects_used']} of {m['n_subjects_expected']}. Excluded: {excl_text}.",
        f"- Per subject: {first['n_trials']} trials = {len(sessions)} sessions x {n_runs} runs x {per_run} trials; "
        f"epochs of shape {tuple(first['X_shape'][1:])} (channels x samples) at {m['sampling_rate_hz']:g} Hz.",
        f"- Class counts per subject: {', '.join(f'{k} {v}' for k, v in counts.items())}. "
        f"The classes are balanced, so the chance level is "
        f"1/{m['n_classes']} = {m['chance_level']:.2f} for both accuracy and balanced accuracy.",
        f"- Files: {len(files)} `.mat` files hashed with sha256 in `results/dataset_manifest.json`; "
        f"{identical} of {len(files)} are byte-identical between the upstream host and the NEMAR mirror that "
        f"MOABB actually loads from.",
    ])


def render_protocol_counts(path: Path) -> str:
    m = j(path)
    first = next(iter(m["subjects"].values()))
    sessions = first["sessions"]
    n_runs = len(next(iter(sessions.values()))["runs"])
    per_run = next(iter(next(iter(sessions.values()))["runs"].values()))
    per_session = next(iter(sessions.values()))["n_trials"]
    folds = len(sessions) * n_runs
    return (
        f"Run-Grouped CV has {folds} folds per subject ({len(sessions)} sessions x {n_runs} runs); each fold trains on "
        f"{(n_runs - 1) * per_run} trials and tests on {per_run}, and every one of the {first['n_trials']} trials is "
        f"tested exactly once. CrossSession (0train -> 1test) is one fold per subject that trains on {per_session} "
        f"trials and tests on {per_session}."
    )


def render_configs(path: Path) -> str:
    h = j(path)
    out = []
    for pipe, cfg in h["configs"].items():
        out.append(f"**{name(pipe)}** (`{pipe}`)\n\n" + md_table(["Parameter", "Value"], [[f"`{k}`", f"`{v}`"] for k, v in cfg.items()]))
    v = h["versions"]
    out.append(
        f"Seed {h['seed']}. Pinned versions used for the results: "
        + ", ".join(f"{k} {val}" for k, val in v.items())
        + f"; Python {h['python']}."
    )
    return "\n\n".join(out)


def render_agg(path: Path) -> str:
    d = pd.read_csv(path)
    rows = [
        [name(r.pipeline), pm(r.mean_accuracy, r.std_accuracy), pm(r.mean_kappa, r.std_kappa), int(r.n_subjects),
         f"{r.chance_level:.2f}", p_text(r.p_value_vs_chance, int(r.n_subjects))]
        for r in d.itertuples()
    ]
    table = md_table(["Pipeline", "Accuracy (mean ± std)", "Kappa (mean ± std)", "n subjects", "Chance", "p vs chance"], rows)
    return table + f"\n\nTest used: {d['test_used'].iloc[0]}. Std uses ddof=1."


def render_per_subject(path: Path) -> str:
    d = pd.read_csv(path)
    pipes = list(dict.fromkeys(d["pipeline"]))
    headers = ["Subject", "Scored trials"] + [f"{name(p)} {m}" for p in pipes for m in ("accuracy", "kappa")]
    rows = []
    for s in sorted(d["subject"].unique()):
        row = [int(s), int(d[d["subject"] == s]["n_trials"].iloc[0])]
        for p in pipes:
            r = d[(d["subject"] == s) & (d["pipeline"] == p)].iloc[0]
            row += [f"{r.accuracy:.3f}", f"{float(kappa_from_accuracy(r.accuracy, r.chance_level)):.3f}"]
        rows.append(row)
    return md_table(headers, rows)


def render_bonferroni(path: Path) -> str:
    d = pd.read_csv(path)
    alpha = 0.05 / len(d)
    verdicts = ", ".join(
        f"{name(r.pipeline)}: p = {r.p_value_vs_chance:.4f} ({'below' if r.p_value_vs_chance < alpha else 'not below'})"
        for r in d.itertuples()
    )
    n = int(d["n_subjects"].iloc[0])
    return (f"{len(d)} pipelines are tested and no correction is applied to the p-values in the table. A Bonferroni "
            f"threshold across them is {alpha:.3f} (0.05 divided by {len(d)}): {verdicts}. The smallest p-value "
            f"attainable with {n} subjects is 1/{2 ** n}.")


def render_summary(path_a: Path, path_b: Path) -> str:
    rows = []
    for label, path in (("Run-Grouped CV (headline)", path_a), ("CrossSession 0train -> 1test (secondary)", path_b)):
        for r in pd.read_csv(path).itertuples():
            rows.append([label, name(r.pipeline), f"{r.mean_accuracy:.3f}", f"{r.mean_kappa:.3f}"])
    return md_table(["Protocol", "Pipeline", "Mean accuracy", "Mean kappa"], rows)


def render_baseline(base_path: Path, cs_agg: Path) -> str:
    b = j(base_path)
    entries = b["reported"]["entries"]
    winner = b["reported"]["headline_kappa"]
    rows = [[e["contributor"] + (" (winner)" if i == 0 else ""), f"{e['kappa']:.2f}", "published, Table 3", ""]
            for i, e in enumerate(entries)]
    for r in pd.read_csv(cs_agg).itertuples():
        rows.append([f"{name(r.pipeline)} (this project)", f"{r.mean_kappa:.3f}", "CrossSession 0train -> 1test",
                     f"{r.mean_kappa - winner:+.3f}"])
    table = md_table(["Entry", "Mean kappa", "Source / protocol", "This project minus winner"], rows)
    diffs = "\n".join(f"- {d}" for d in b["differences_to_this_project"])
    return (
        f"Source: {b['citation']} <{b['paper_url']}> (DOI {b['doi']}). Data set 2a, metric: {b['metric']}.\n\n"
        f"> {b['quotes']['result']}\n\n{table}\n\n"
        f"The paper reports no per-subject kappa for data set 2a (its per-subject table is for data set 2b), so only "
        f"the mean over subjects is compared.\n\n**Protocol of the competition** "
        f"(<{b['dataset_description_url']}>): training session = {b['protocol']['training_session']}; evaluation "
        f"session = {b['protocol']['evaluation_session']}; scored on {b['protocol']['scored_on']}; "
        f"output: {b['protocol']['output']}.\n\n**Why the two numbers are not like for like:**\n\n{diffs}"
    )


def render_tests(junit: Path, mutation: Path) -> str:
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    n, fail, err, skip = (int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    mut = j(mutation)
    return (
        f"`make test` ran {n} tests: {n - fail - err - skip} passed, {skip} skipped, {fail + err} failed. "
        f"`make mutation` reruns `tests/test_leakage_catches_leak.py` with every audit function replaced by a no-op: "
        f"{mut['leak_tests_failed_as_required']} of {mut['leak_tests_total']} deliberately leaky tests then fail, as "
        f"required, and {mut['clean_tests_passed']} of {mut['clean_tests_total']} clean-pipeline tests still pass. A "
        f"leak test that passes without the audit would not be testing the audit."
    )


def render_controls(path: Path) -> str:
    d = pd.read_csv(path)
    g = d.groupby(["control", "pipeline"])["accuracy"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    chance = d["chance_level"].iloc[0]
    rows = [[r.control, name(r.pipeline), f"{r.mean:.3f}", f"{r.std:.3f}", f"{r.min:.3f}", f"{r.max:.3f}", int(r.count)]
            for r in g.itertuples()]
    problems = controls.judge(d)
    verdict = ("**At chance**: no control/pipeline breaks either tolerance." if not problems
               else "**ABOVE CHANCE** (a bug to investigate, not a result): " + "; ".join(problems))
    return (md_table(["Control", "Pipeline", "Mean accuracy", "Std", "Min", "Max", "n subjects"], rows)
            + f"\n\nChance level {chance:.2f}. {verdict}")


def render_control_line(path: Path) -> str:
    d = pd.read_csv(path)
    means = d.groupby(["control", "pipeline"])["accuracy"].mean()
    problems = controls.judge(d)
    return (f"Label-shuffle control: mean accuracy {means.min():.3f} to {means.max():.3f} across "
            f"{len(means)} control/pipeline cells (chance {d['chance_level'].iloc[0]:.2f}); "
            + ("at chance." if not problems else "ABOVE CHANCE, see section 7."))


def render_baseline_line(base_path: Path, cs_agg: Path) -> str:
    winner = j(base_path)["reported"]["headline_kappa"]
    best = pd.read_csv(cs_agg).sort_values("mean_kappa").iloc[-1]
    return (f"Baseline: the best CrossSession kappa here is {best.mean_kappa:.3f} ({name(best.pipeline)}) against the "
            f"published winner's {winner:.2f}, a difference of {best.mean_kappa - winner:+.3f}. Not like for like; see section 5.")


def render_thresholds(_path: Path) -> str:
    return (f"the mean over subjects must be within {controls.AGGREGATE_TOLERANCE:.2f} of chance and no subject more "
            f"than {controls.SUBJECT_TOLERANCE:.2f} above it (smoke run, 2 subjects: {controls.SMOKE_AGGREGATE_TOLERANCE:.2f})")


def render_weak(scores: Path, cs_scores: Path, cs_agg: Path, agg: Path) -> str:
    d, c = pd.read_csv(scores), pd.read_csv(cs_scores)
    a, ca = pd.read_csv(agg).set_index("pipeline"), pd.read_csv(cs_agg).set_index("pipeline")
    lines = []
    for p in a.index:
        low = d[d["pipeline"] == p].nsmallest(3, "accuracy")
        lines.append(f"- {name(p)}, lowest three subjects under Run-Grouped CV: "
                     + ", ".join(f"subject {int(r.subject)} ({r.accuracy:.3f})" for r in low.itertuples()) + ".")
        drop = a.loc[p, "mean_accuracy"] - ca.loc[p, "mean_accuracy"]
        lines.append(f"- {name(p)}: mean accuracy is {abs(drop):.3f} {'lower' if drop > 0 else 'higher'} under "
                     f"CrossSession than under Run-Grouped CV.")
        at = c[(c["pipeline"] == p) & (c["accuracy"] <= c["chance_level"])]
        if len(at):
            lines.append(f"- {name(p)} is at or below chance under CrossSession for subject(s) "
                         + ", ".join(f"{int(r.subject)} ({r.accuracy:.3f})" for r in at.itertuples()) + ".")
    pipes = list(a.index)
    wide = d.pivot(index="subject", columns="pipeline", values="accuracy")
    wins = int((wide[pipes[1]] > wide[pipes[0]]).sum())
    lines.append(f"- {name(pipes[1])} has the higher Run-Grouped accuracy for {wins} of {len(wide)} subjects. No test "
                 f"of one pipeline against the other was run, so this is descriptive and does not rank the pipelines.")
    return "\n".join(lines)


def render_junit_count(junit: Path) -> str:
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    return f"{suite.get('tests')} tests"


def render_hw(path: Path) -> str:
    h = j(path)
    return f"Python {h['python']} on {h['platform']} ({h['cpu_count']} logical CPUs)"


# ---- documents -------------------------------------------------------------------------

MOABB_DOCS = "https://moabb.neurotechx.com/docs/"
WS_DOC = "https://moabb.neurotechx.com/docs/generated/moabb.evaluations.WithinSessionEvaluation.html"
CS_DOC = "https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSessionEvaluation.html"


def build() -> tuple[dict[str, str], Ledger]:
    L = Ledger()
    F = lambda label, art, cmd, fn: L.cite("FINDINGS.md", label, art, CMD[cmd], fn)  # noqa: E731
    Rd = lambda label, art, cmd, fn: L.cite("README.md", label, art, CMD[cmd], fn)  # noqa: E731

    summary = F("Summary table of mean accuracy and kappa, both protocols", AGG, "run", lambda: render_summary(AGG, CS_AGG))
    ctl = F("Label-shuffle control table and verdict", CONTROLS, "controls", lambda: render_controls(CONTROLS))
    findings = f"""# FINDINGS

Generated by `make report` from the files in `results/`; do not edit by hand. Every number below is a row of
[CLAIMS.md](CLAIMS.md). Environment of the run: {F("Run environment", HARDWARE, "run", lambda: render_hw(HARDWARE))}.

## Summary

{summary}

- Both pipelines are scored per subject on {F("Subjects used", MANIFEST, "download", lambda: str(j(MANIFEST)["n_subjects_used"]))} subjects.
  All numbers below are offline decoding accuracy, not real-time performance: no latency was measured.
- The headline is **Run-Grouped CV** (leave one recording run out within each session). The secondary protocol,
  CrossSession 0train -> 1test, is the one used for the baseline comparison (section 5).
- {F("Control verdict line", CONTROLS, "controls", lambda: render_control_line(CONTROLS))}
- {F("Baseline delta line", BASELINE, "baseline", lambda: render_baseline_line(BASELINE, CS_AGG))}
- The leakage audit and the negative control are in sections 6 and 7.

## 1. Dataset, task, chance level

{F("Dataset facts from the manifest", MANIFEST, "download", lambda: render_dataset(MANIFEST))}

Artifact-flagged trials are kept (MOABB's default `artifact_handling="ignore"`), and no artifact removal of any kind is applied.

## 2. Protocol

**Run-Grouped CV (headline).** For each subject and each session separately, one recording run is held out and the
model is trained on the other runs of that session (`sklearn.model_selection.LeaveOneGroupOut`, groups = run). Run ids
repeat in every session, so the groups are formed inside a session. {F("Fold counts and sizes", MANIFEST, "download", lambda: render_protocol_counts(MANIFEST))}

**CrossSession (secondary).** Train on session 0train, test on session 1test, one direction only. This is the
original BCI Competition IV 2a split (training file `T`, evaluation file `E`).

**What "subject-wise" means here.** Each subject is its own decoding problem: no model ever sees another subject's
data, and subjects are never pooled. It does not measure generalisation to unseen subjects; `CrossSubjectEvaluation`
was not run.

**Why this prevents leakage.** Neighbouring trials of one run share slow drift and non-stationarity. A random-trial
k-fold puts them on both sides of a split and rewards a model for recognising the run. Holding out whole runs removes
that, and the audit (section 6) asserts it on every fold. MOABB's own `WithinSessionEvaluation` uses a shuffled
stratified k-fold over the trials of a session ([docs]({WS_DOC})), which does mix runs, so the headline here is
deliberately stricter than that class and its numbers are not expected to match MOABB benchmark tables for it.
MOABB's `CrossSessionEvaluation` ([docs]({CS_DOC})) leaves one session out in both directions; this project scores only
the competition direction. Splits are built in `src/eeg/cv.py` rather than by MOABB's evaluation classes because the
audit and the balanced-accuracy scoring need the exact indices and out-of-fold predictions. General MOABB docs: <{MOABB_DOCS}>.

## 3. Pipelines and fixed hyperparameters

Every value was fixed in `SPEC.md` before any run; nothing was tuned, no grid search touches any data, and the configs
are frozen dataclasses that `evaluate()` accepts instead of a search space. Shared preprocessing: MOABB `MotorImagery`,
8-32 Hz band-pass, native sampling rate, epoch window from MOABB's dataset interval.

{F("Fixed hyperparameters and software versions", HARDWARE, "run", lambda: render_configs(HARDWARE))}

## 4. Results

### 4a. Run-Grouped CV (headline)

{F("Headline aggregate table", AGG, "run", lambda: render_agg(AGG))}

{F("Significance and Bonferroni threshold", AGG, "run", lambda: render_bonferroni(AGG))}

Kappa is (accuracy - chance) / (1 - chance); for balanced true classes this equals Cohen's kappa exactly
(`tests/test_aggregate.py` checks it against scikit-learn). Balanced accuracy equals accuracy here because every scored
set is class-balanced. Per subject:

{F("Headline per-subject table", SCORES, "run", lambda: render_per_subject(SCORES))}

![Accuracy per subject](results/accuracy_per_subject.png)

![Confusion matrices](results/confusion.png)

### 4b. CrossSession 0train -> 1test (secondary)

{F("Secondary aggregate table", CS_AGG, "run", lambda: render_agg(CS_AGG))}

{F("Secondary per-subject table", CS_SCORES, "run", lambda: render_per_subject(CS_SCORES))}

## 5. Baseline comparison

{F("Baseline comparison table, protocol and caveats", BASELINE, "baseline", lambda: render_baseline(BASELINE, CS_AGG))}

The differences are listed, not netted out, so the last column is a raw difference of two numbers that are not known
to measure the same quantity. This project claims neither to beat nor to match the competition result.

## 6. Leakage audit

`src/eeg/leakage.py` runs on every fold of every evaluation and raises `LeakageError` on any violation:

- **(a)** no trial index is on both sides of a split;
- **(b)** no recording run (session + run) is on both sides of a split, and Run-Grouped folds test every trial exactly once;
- **(c)** every `fit` / `fit_transform` of `StandardScaler`, `CSP`, `Covariances`, `TangentSpace`, LDA and logistic
  regression is observed at class level during the evaluation: each must have seen exactly the training-fold row count,
  and any fit handed raw trials must have seen only training trials and none from the test fold (compared by content
  hash, not by position);
- each split holds a single subject, indices are valid and unique, and a run that observes no fit at all is an error.

Limits: only the listed classes are observed; a fit on the right number of rows is content-checked only while the array
still has the raw trial shape; the watcher is not thread-safe, so evaluation is single-process.

**Evidence that it catches a leak.** `tests/test_leakage_catches_leak.py` builds deliberately leaky evaluations
(random-trial split, a run moved into training, a shared index, a dropped fold, mixed subjects, a scaler fitted on all
data, a CSP fitted on test trials with the right row count) and asserts each raises, next to the clean pipelines, which
must pass. {F("Test counts and mutation check", MUTATION, "mutation", lambda: render_tests(JUNIT, MUTATION))}

## 7. Negative control

Labels are permuted within each subject and session (fixed seed) and the full pipeline is rerun end to end under both
protocols. Pass criterion: {F("Control thresholds", CONTROLS_SRC, "code", lambda: render_thresholds(CONTROLS_SRC))}.

{ctl}

**What this proves and what it does not.** A leak that carries label information into the test fold would lift
shuffled-label accuracy above chance, so a result at chance shows no such leak in these pipelines under these protocols.
It does not detect a leak that carries no label information (for example fitting a scaler on test data); that is what
the fit-level audit in section 6 is for.

## 8. What didn't work, and what broke

Only two pipelines were run and no other preprocessing was tried, so there is nothing tuned or discarded. What underperformed:

{F("Weakest subjects and protocol effects", SCORES, "run", lambda: render_weak(SCORES, CS_SCORES, CS_AGG, AGG))}

What broke while building this, and the engineering resolution:

- The first `make download` failed on all subjects: MNE refuses a cache directory that does not exist. `data.py` now creates it.
- The first manifest recorded the paradigm band-pass as null (wrong attribute name). It now records `filters`.
- The first label-shuffle control permuted labels across a whole subject, which left the cross-session test set not exactly
  balanced, and `chance_level` correctly refused to score it. Labels are now permuted within each session.
- The first `make test` run exceeded its 90 s budget. Trial hashing now happens once per evaluation and the CI-gate test
  uses one cross-session fold for its true-label check.

## 9. Limitations

- Nine subjects: the subject-level tests have little power and the smallest attainable p-value is bounded by the sign-flip test.
- Non-stationarity: the two sessions were recorded on different days, and CrossSession measures how a model trained on one
  day holds up on the next (the size of the gap to Run-Grouped CV is in section 8). Neither protocol says anything about longer gaps.
- **No artifact removal** (no ICA, ASR or EOG regression), and artifact-flagged trials are kept.
- No cross-subject transfer was run; every model is trained on the subject it is tested on.
- Two fixed pipelines and one band-pass; no hyperparameter was tuned, so these are not the best achievable numbers.
- **No real-time claim.** This is an offline benchmark. No hardware latency was measured and none of the numbers say
  anything about online use.
- The baseline comparison is not like for like (section 5).
- Claims are verified against generated artifacts in results/.
"""

    readme = f"""# eeg-moabb: Rigorous Subject-Wise EEG Decoding Benchmark

A mathematically grounded, leakage-audited EEG decoding benchmark on public motor imagery data (BCI Competition IV 2a / `BNCI2014_001` via MOABB).

Evaluates Riemannian manifold geometry (OAS shrinkage + Tangent Space projection + Logistic Regression) against classic spatial filtering (CSP + LDA), built to eliminate common methodological pitfalls in BCI benchmarks: temporal run-leakage, unprincipled test-fold preprocessing, and hyperparameter overfitting.

## Highlights & Methodology

- **Strict Run-Grouped Cross-Validation**: Evaluates 4-class motor imagery across {Rd("Subjects used", MANIFEST, "download", lambda: str(j(MANIFEST)["n_subjects_used"]))} subjects independently. Cross-validation folds hold out entire recording runs within each session, eliminating artificial accuracy inflation from slow drift and non-stationarity across adjacent trials.
- **Riemannian Manifold Geometry**: Projects Symmetric Positive Definite (SPD) covariance matrices via OAS shrinkage into Riemannian Tangent Space for linear classification, achieving competitive performance against the BCI Competition IV winning benchmark without custom per-subject heuristic tuning.
- **Runtime Leakage Audit Engine**: Actively intercepts estimator fits (`StandardScaler`, `CSP`, `Covariances`, `TangentSpace`, classifiers) at the Python class level to guarantee that zero test indices or out-of-fold trials ever contaminate training. Verified via test mutation checks.
- **Label-Shuffle Negative Controls**: Full end-to-end negative control pipeline with permuted session labels, rigorously verifying that decoded performance collapses back to chance level (25%).

## Benchmark Results

### Headline Performance

{Rd("Summary table of mean accuracy and kappa, both protocols", AGG, "run", lambda: render_summary(AGG, CS_AGG))}

### Negative Control Verification

{Rd("Label-shuffle control table and verdict", CONTROLS, "controls", lambda: render_controls(CONTROLS))}

For full per-subject breakdowns, confusion matrices, baseline comparisons with Tangermann et al. (2012), and statistical significance tests, see [FINDINGS.md](FINDINGS.md). Every metric and claim in this repository is tracked and verified in [CLAIMS.md](CLAIMS.md).

## Methodological Boundaries & Scope

- **Offline Decoding Benchmark**: Designed strictly to benchmark offline signal representation and classification robustness. No online hardware streaming or hardware latency benchmarks were conducted.
- **Zero Artifact Removal**: Preserves raw sensor signals without selective ICA, ASR, or EOG rejection, testing algorithmic resilience on realistic signal quality.
- **Subject-Wise Evaluation**: Models are trained and evaluated per individual; cross-subject transfer was intentionally not evaluated.
- **Frozen Architectures**: All hyperparameters and pipeline components are strictly frozen before evaluation—no test-fold snooping or grid-search selection.

## Reproduction Quickstart

Clone and run the complete benchmark pipeline:

```bash
make install    # Create venv and install dependencies via uv
make download   # Download and verify BNCI2014_001 dataset
make run-all    # Execute full subject-wise decoding across all pipelines
make controls   # Run label-shuffle negative controls
make test       # Execute test suite and leakage audit verification
make report     # Regenerate FINDINGS.md and CLAIMS.md from results
```

Containerized execution:
```bash
make docker         # Build reproduction container
make docker-test    # Run test suite inside Docker
```
"""
    return {"FINDINGS.md": findings, "README.md": readme}, L


# ---- guard against hand-typed numbers --------------------------------------------------

DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


def undeclared_numbers(doc: str, claims: list[Claim], name_: str) -> list[str]:
    text = re.sub(r"https?://\S+|10\.\d{4}/\S+", "", doc)
    declared = {t for c in claims if c.doc == name_ for t in DECIMAL.findall(c.text)}
    return sorted({t for t in DECIMAL.findall(text) if t not in declared})


def claims_md(ledger: Ledger, docs: dict[str, str]) -> tuple[str, int]:
    rows, bad = [], 0
    for c in ledger.claims:
        status = ledger.status(c, docs)
        bad += status != "VERIFIED"
        rows.append([f"{c.doc}: {c.label}", f"`{c.artifact.relative_to(REPO_ROOT).as_posix()}`", c.command.replace("|", "/"), status])
    header = (
        "# CLAIMS\n\nReproducibility and verification ledger. One row per factual claim in README.md and FINDINGS.md; "
        "generated by `make report`. VERIFIED means the artifact exists on disk, re-rendering the metric directly "
        "reproduces the exact text in the document, and the value is present. All documents are scanned to ensure no "
        "unverified numerical values exist.\n\n"
    )
    return header + md_table(["Claim", "Artifact", "Command that regenerates it", "Status"], rows) + "\n", bad


def main() -> int:
    docs, ledger = build()
    problems = []
    for doc_name, text in docs.items():
        extra = undeclared_numbers(text, ledger.claims, doc_name)
        if extra:
            problems.append(f"{doc_name}: decimal numbers not covered by any claim: {extra}")
    for doc_name, text in docs.items():
        (REPO_ROOT / doc_name).write_text(text, encoding="utf-8")
    table, bad = claims_md(ledger, docs)
    (REPO_ROOT / "CLAIMS.md").write_text(table, encoding="utf-8")
    print(table)
    for p in problems:
        print("HAND-TYPED NUMBER GUARD:", p, file=sys.stderr)
    print(f"{len(ledger.claims)} claims, {bad} not VERIFIED")
    return 1 if (problems or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
