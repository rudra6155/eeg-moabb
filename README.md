# eeg-moabb: Rigorous Subject-Wise EEG Decoding Benchmark

A mathematically grounded, leakage-audited EEG decoding benchmark on public motor imagery data (BCI Competition IV 2a / `BNCI2014_001` via MOABB).

Evaluates Riemannian manifold geometry (OAS shrinkage + Tangent Space projection + Logistic Regression) against classic spatial filtering (CSP + LDA), built to eliminate common methodological pitfalls in BCI benchmarks: temporal run-leakage, unprincipled test-fold preprocessing, and hyperparameter overfitting.

## Highlights & Methodology

- **Strict Run-Grouped Cross-Validation**: Evaluates 4-class motor imagery across 9 subjects independently. Cross-validation folds hold out entire recording runs within each session, eliminating artificial accuracy inflation from slow drift and non-stationarity across adjacent trials.
- **Riemannian Manifold Geometry**: Projects Symmetric Positive Definite (SPD) covariance matrices via OAS shrinkage into Riemannian Tangent Space for linear classification, achieving competitive performance against the BCI Competition IV winning benchmark without custom per-subject heuristic tuning.
- **Runtime Leakage Audit Engine**: Actively intercepts estimator fits (`StandardScaler`, `CSP`, `Covariances`, `TangentSpace`, classifiers) at the Python class level to guarantee that zero test indices or out-of-fold trials ever contaminate training. Verified via test mutation checks.
- **Label-Shuffle Negative Controls**: Full end-to-end negative control pipeline with permuted session labels, rigorously verifying that decoded performance collapses back to chance level (25%).

## Benchmark Results

### Headline Performance

| Protocol | Pipeline | Mean accuracy | Mean kappa |
|---|---|---|---|
| Run-Grouped CV (headline) | CSP + LDA | 0.618 | 0.490 |
| Run-Grouped CV (headline) | Riemannian tangent space + LR | 0.725 | 0.633 |
| CrossSession 0train -> 1test (secondary) | CSP + LDA | 0.551 | 0.402 |
| CrossSession 0train -> 1test (secondary) | Riemannian tangent space + LR | 0.670 | 0.560 |

### Negative Control Verification

| Control | Pipeline | Mean accuracy | Std | Min | Max | n subjects |
|---|---|---|---|---|---|---|
| label_shuffle | CSP + LDA | 0.248 | 0.019 | 0.224 | 0.285 | 9 |
| label_shuffle | Riemannian tangent space + LR | 0.248 | 0.024 | 0.219 | 0.295 | 9 |
| label_shuffle_cross_session | CSP + LDA | 0.253 | 0.018 | 0.219 | 0.278 | 9 |
| label_shuffle_cross_session | Riemannian tangent space + LR | 0.259 | 0.020 | 0.236 | 0.306 | 9 |

Chance level 0.25. **At chance**: no control/pipeline breaks either tolerance.

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
