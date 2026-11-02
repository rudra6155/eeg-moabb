# SPEC

Scope specification for the benchmark. Written before any pipeline or evaluation code. Dataset facts were first written as **PENDING** (expected values from the dataset documentation) and were checked against `results/dataset_manifest.json` after `make download` on 2026-10-29; their status is now **CONFIRMED**. Later edits changed status labels, recorded facts the download revealed ("added after download") and recorded the evaluation decisions in section 3. No hyperparameter was changed after seeing a result.

## 1. Dataset

| Field | Value | Status |
|---|---|---|
| Name | `BNCI2014_001` (BCI Competition IV dataset 2a), `moabb.datasets.BNCI2014_001`; MOABB's internal dataset code is `BNCI2014-001` (hyphen) | fixed |
| Version | MOABB `1.7.2` (the dataset has no version number of its own; the MOABB release pins the loader and download URL). File identity is pinned by the sha256 values in `results/dataset_manifest.json`. | fixed |
| n_subjects | 9 | CONFIRMED (9/9 loaded, none excluded) |
| Sessions per subject | 2 (`0train`, `1test`), recorded on different days | CONFIRMED (two sessions, 288 trials each) |
| Runs per session | 6 | CONFIRMED |
| Trials per run | 48 (12 per class) | CONFIRMED (every run of every subject: 48) |
| n_trials_per_subject | 576 (2 sessions x 6 runs x 48) | CONFIRMED (all 9 subjects) |
| n_classes | 4 (`left_hand`, `right_hand`, `feet`, `tongue`) | CONFIRMED |
| sampling_rate | 250 Hz | CONFIRMED |
| EEG channels | 22 (the 3 EOG channels are not in the epoched data) | CONFIRMED (`X.shape == (576, 22, 1001)`) |
| chance_level | 1/n_classes = 0.25 | CONFIRMED: class counts are exactly 144 per class for every subject (max class fraction 0.25), so the balanced and plain chance levels coincide; a test asserts this against the manifest |

Alternative considered and not chosen: `PhysionetMI`. One dataset only.

Added after download:

- **Data source.** `data_path()` points at the upstream host (`https://lampx.tugraz.at/~bci/database/001-2014/`), but in MOABB 1.7.2 `get_data()` loads from a NEMAR mirror (`nm000139`, `sourcedata/`) when it exists. All 18 `.mat` files (about 780 MB) were hashed from both locations and are byte-identical, so the manifest hashes describe the data that is loaded. `make download` re-checks this every time and fails if the copies ever differ.
- **Artifact-flagged trials are kept.** The original dataset flags some trials as artifacts. MOABB's default `artifact_handling="ignore"` keeps them, and this project does not change that. This is part of the "no artifact removal" scope decision and must be stated in FINDINGS.md.
- **Epoch length.** Epochs come from MOABB's dataset interval `[2, 6]` s, which is 4 s or 1001 samples at 250 Hz.

## 2. Task

4-class cue-based motor imagery: **left hand / right hand / both feet / tongue**. Epoch window and band-pass come from the paradigm settings in section 4; the window is MOABB's default for this dataset and is not tuned.

## 3. Evaluation

Everything is strictly per subject: one decoding problem per subject, subjects never pooled.

| Role | Protocol | Folds | Scored trials per subject |
|---|---|---|---|
| **Headline** | **Run-Grouped CV**: within each session, leave one recording run out (`sklearn.model_selection.LeaveOneGroupOut`, groups = run) | 12 (2 sessions x 6 runs); train 240, test 48 | 576 (every trial tested exactly once) |
| Secondary | **CrossSession**: train on `0train`, test on `1test` (one direction) | 1; train 288, test 288 | 288 |

FINDINGS.md must label the headline "Run-Grouped CV" and the secondary "CrossSession (0train -> 1test)".

MOABB has no `WithinSubjectEvaluation`; the three real classes are `WithinSessionEvaluation`, `CrossSessionEvaluation` and `CrossSubjectEvaluation`. Docs: <https://moabb.neurotechx.com/docs/>.

### Why these two, and why not MOABB's classes (decision, 2026-10-29)

- **Audit check (b) forces run-grouped folds.** MOABB's `WithinSessionEvaluation` uses stratified k-fold over the trials of a session, which puts trials of one run on both sides of a split. Neighbouring trials share drift and slow non-stationarity, so that protocol can inflate accuracy. Leaving whole runs out removes that and lets the audit assert (b) on the headline instead of relaxing it. The headline is therefore stricter than MOABB's `WithinSessionEvaluation`, and its numbers are not expected to match MOABB benchmark tables for that class.
- **Run ids repeat across sessions** (`0`..`5` in both), so groups are formed inside each session. Grouping on the run id alone would pool two different physical runs.
- **The original BCI Competition IV 2a protocol is train on `0train`, test on `1test`**, which is what the secondary protocol does. The baseline comparison uses this protocol, not the headline. MOABB's `CrossSessionEvaluation` leaves one session out in both directions; this project scores only the competition direction so that the comparison is like for like.
- **Own splits, not MOABB's evaluation classes.** The leakage audit, the balanced accuracy and the confusion matrices need the exact train/test indices and out-of-fold predictions. MOABB's classes return aggregated scores. `src/eeg/cv.py` therefore builds the splits and `src/eeg/evaluate.py` runs them. The data loading and epoching still go through MOABB's `MotorImagery` paradigm.
- **Not run:** `CrossSubjectEvaluation`. "Subject-wise" here means each subject is its own decoding problem. It does not mean generalisation to unseen subjects, and FINDINGS.md must not imply that it does.

Metric: accuracy and balanced accuracy over all out-of-fold trials of a subject. `n_trials` in the score tables is the number of scored trials (576 headline, 288 secondary).

**Baseline, resolved 2026-10-29 from the sources, which were read in full text:** the competition ranked entries by Cohen's kappa on the evaluation session (`E`), trained on the labelled session (`T`), scored on artifact-free trials from a causal per-sample output. The winner's mean kappa over the nine subjects was 0.57 (Tangermann et al. 2012, Table 3). Neither the paper nor the dataset description says how the per-time-point kappa was reduced to one number, and the paper gives no per-subject kappa for data set 2a. Kappa is therefore reported next to accuracy (section 6) and the comparison uses the mean only, on the CrossSession protocol, with every protocol difference listed. The transcribed figures, quotes and differences are in `results/baseline.json`.

## 4. Pipelines and fixed hyperparameters

All values are fixed before any run. No grid search, no value is chosen by looking at test-fold results. Any argument not listed is left at the library default of the pinned version.

**Shared preprocessing** (`moabb.paradigms.MotorImagery`, `src/eeg/data.py`):

| Setting | Value |
|---|---|
| `n_classes` | 4 |
| band-pass `fmin`-`fmax` | 8-32 Hz (`filters=[[8, 32]]`, which is also MOABB's default) |
| `tmin`, `tmax` | `0.0`, `None`: MOABB's dataset interval, not overridden |
| `resample` | none (native 250 Hz) |
| Artifact removal | none |

**Pipeline 1: CSP + LDA** (`src/eeg/csp_lda.py`, name `csp_lda`)

| Component | Setting |
|---|---|
| `mne.decoding.CSP` | `n_components=4`, `reg=None`, `log=True`, `norm_trace=False` |
| `LinearDiscriminantAnalysis` | `solver="svd"`, `shrinkage=None` |
| Scaling | none |

**Pipeline 2: Riemannian tangent space + logistic regression** (`src/eeg/riemann.py`, name `riemann_ts_lr`)

| Component | Setting |
|---|---|
| `pyriemann.estimation.Covariances` | `estimator="oas"` (fixed shrinkage, not tuned) |
| `pyriemann.tangentspace.TangentSpace` | `metric="riemann"`, `tsupdate=False` (default); the reference covariance is fitted on the training fold only |
| `LogisticRegression` | `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`. The penalty is L2: scikit-learn 1.9.1 deprecates the `penalty` argument and its default `l1_ratio=0.0` means L2. |
| Scaling | none |

**Seed:** `42` for python, numpy (scikit-learn and pyriemann draw from it) and the one estimator that takes a `random_state`. Neither pipeline has a stochastic component (closed-form CSP and LDA, deterministic lbfgs, fixed folds), so the seed cannot change their accuracy; a test asserts exactly that. The seed matters only for the label-shuffle control.

**Configs are frozen dataclasses of single values** (`CspLdaConfig`, `RiemannConfig`). `evaluate()` accepts only these; a dict, a list of candidates, a config holding a list, or a `GridSearchCV` raises `TypeError`.

**No deep learning.** This is a deliberate scope decision, not a gap: the question here is how honest the evaluation is, not how high the number can go.

## 5. Leakage audit and negative control

**Audit** (`src/eeg/leakage.py`), run on every split and every fold of every evaluation, raising `LeakageError` on a violation:

- Split level: (a) no trial index on both sides; (b) no recording run (session + run) on both sides; indices valid and unique; a single subject per split; run-grouped folds test every trial exactly once.
- Fit level, check (c): while an evaluation runs, `fit` and `fit_transform` of `StandardScaler`, `CSP`, `Covariances`, `TangentSpace`, `LinearDiscriminantAnalysis` and `LogisticRegression` are observed at class level. Each fit must have seen exactly as many rows as the training fold, and any fit handed raw trials must have seen only training trials (compared by content hash, not position).
- Limits, to be repeated in FINDINGS.md: only those classes are observed; a fit on the right number of rows is only content-checked while the array still has the raw trial shape; the watcher is not thread-safe, so the evaluation is single-process.

**Label-shuffle control** (`src/eeg/controls.py`): the full pipeline is run end to end with labels permuted within each subject, fixed seed, for both pipelines and both protocols. Labels are permuted among the trials of each session separately, not across the whole subject, so each session keeps exactly balanced classes and the cross-session test set (session `1test` alone) still has chance level exactly 1/n_classes. Full run: the mean over subjects must be within 0.03 of chance, and no subject more than 0.10 above it. Smoke run (2 subjects): 0.10 for both, because 0.03 would be only about 2 standard errors there. Failure writes `results/leakage_investigation.txt` and exits non-zero.

## 6. Statistics

- Per subject: accuracy, balanced accuracy, n_trials, chance level.
- Aggregate per pipeline: mean and std (ddof=1) over subjects, n_subjects.
- **Kappa** (added 2026-10-29): per subject `(accuracy - chance) / (1 - chance)`; the aggregate reports its mean and std (ddof=1) as `mean_kappa` and `std_kappa`. For balanced true classes this is exactly Cohen's kappa, because the expected agreement is 1/K whatever the predictions are; a test checks it against scikit-learn's `cohen_kappa_score`. The per-subject CSV keeps its original columns (kappa is derived from them). The p-value stays on accuracy; it would be the same test on kappa, since kappa is a monotone function of accuracy.
- Significance: one-sided exact sign-flip permutation test on the subject-level values `accuracy - chance`, with the subject as the unit. Pooling trials across subjects would ignore between-subject variance and give p-values that are too small. With n=9 the smallest attainable p is 1/512.
- Multiplicity: none applied to the p-values in `aggregate.csv`. Two pipelines are tested, so FINDINGS.md will also state the Bonferroni threshold (alpha = 0.025).
- Balanced accuracy is always reported; class counts are identical across subjects (144 each) and balanced in every scored set, so balanced accuracy will equal plain accuracy exactly. FINDINGS.md states the check.
- The aggregation fails hard on a missing, duplicated, unexpected or malformed per-subject row. A subject whose pipeline fails stays in the denominator and is logged to `pipeline_failures.txt`.

## 7. Outputs

Headline files in `results/`: `per_subject_scores.csv`, `aggregate.csv`, `accuracy_per_subject.png`, `confusion.png`, `hardware.json`, `excluded_subjects.txt`, plus `pipeline_failures.txt` only if something failed. Secondary files in `results/cross_session/` (`per_subject_scores.csv`, `aggregate.csv`, same schemas). `results/controls.csv` holds both control variants (`label_shuffle`, `label_shuffle_cross_session`). The smoke run writes the same layout to `results_smoke/`.

Schema amendment: `aggregate.csv` is `pipeline, mean_accuracy, std_accuracy, mean_kappa, std_kappa, n_subjects, chance_level, p_value_vs_chance, test_used`. The two kappa columns were added on 2026-10-29 and extend the original baseline schema; the per-subject and controls schemas are unchanged.

Evidence files: `results/baseline.json` (hand-transcribed from the cited sources), `results/test_results.xml` (`make test`), `results/leakage_mutation_check.json` (`make mutation`), `results/docker_build.txt` (`make docker`). `FINDINGS.md` and `README.md` are written by hand from these files.

## 8. Out of scope

Stated so they are not read as bugs:

- Artifact removal (ICA / ASR)
- Hyperparameter search of any kind
- EEGNet or any deep model
- Cross-dataset transfer
- Cross-subject transfer (`CrossSubjectEvaluation`)
- Real-time or streaming decoding. No latency measurement exists, so no "real-time" or "online" claim is made anywhere.
- Hardware

## 9. Environment

Python 3.12 in a project-local `.venv`, created by `make install`. Direct dependencies pinned in `pyproject.toml`: `moabb==1.7.2`, `mne==1.13.2`, `pyriemann==0.12`, `scikit-learn==1.9.1`, `numpy==2.5.4`. These are the versions `uv` resolves together for `moabb==1.7.2` on 2026-10-29. The dataset cache is project-local (`data/`, via `MNE_DATA`) and is not committed.
