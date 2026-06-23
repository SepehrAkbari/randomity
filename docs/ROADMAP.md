# Roadmap

Living checklist toward a methodology + software publication. Status:
☐ todo · ◐ in progress · ☑ done. Keep `CLAUDE.md` and `docs/METHODOLOGY.md` in sync.

## Workstream A — Correctness fixes (do first) ☑ DONE (2026-06-23)

- ☑ A1 Permutation test: now a von-Neumann successive-difference permutation test
  (order-sensitive). `_utils/_tests/permute_tst.py`
- ☑ A2 Gap test: geometric null with tail pooling. `_utils/_tests/gap_tst.py`
- ☑ A3 KS test: discrete KS statistic with Monte-Carlo null. `_utils/_tests/ks_tst.py`
- ☑ A4 Entropy: per-sequence `entropy_ratio = H / log₂ k`.
  `_utils/_tests/entropy_tst.py`
- ☑ A5 Serial autocorrelation: emits calibrated `serial_p`; scoring uses it (rewards
  `|ρ| ≈ 0`). `_utils/_tests/serial_tst.py`, `_utils/compute_score.py`
- ☑ A6 FFT: power-normalized periodogram + Fisher's g-test p-value (`fft_g_p`).
  `_utils/_tests/ftt_tst.py`
- ☑ A7 `freq_test` is now an order-2 serial (pair) frequency test, distinct from
  `chisqr_test`. `_utils/_tests/freq_tst.py`
- ☑ Side fix: qiskit is now a lazy import (evaluation works without the quantum stack);
  version lookup falls back when run from source. Tests added under
  `tests/test_evaluate/test_stat_tests.py` (15 passing).

## Workstream B — Principled scoring redesign

- ☑ (partial) `compute_score.py` consumes the corrected, scale-invariant outputs.
- ☐ **PRIORITY**: the averaging aggregation + fixed 0.6 threshold is miscalibrated —
  under H₀ each p-value is U(0,1) so the mean ≈ 0.5, and genuinely random sequences
  hover at/under 0.6 (verified: a uniform n=1000 sequence scored exactly 0.60).
  Replace averaging with Fisher's method per paradigm + overall.
- ☐ Empirically calibrate the combined null (tests are dependent).
- ☐ Multiple-testing control (documented α / BH-FDR) + reported FPR.
- ☐ Every test returns a calibrated p-value under H₀ (entropy still a ratio, needs a
  p-value form for Fisher).
- ☐ Keep `isRandom`/`inspectRandom` API stable; implement `whyRandom` per-test report.

## Workstream C — Predictability dimension (PRNG-vs-TRNG signal)

- ☐ Compression-ratio test (zlib/bz2).
- ☐ Maurer's universal test.
- ☐ Linear complexity / Berlekamp–Massey.
- ☐ Learned next-symbol predictor (n-gram / logistic / MLP) with binomial-test p-value.
- ☐ Report predictability separately from quality.

## Workstream D — Validation against standard batteries

- ☐ Interface NIST STS, Dieharder, TestU01 (SmallCrush/Crush) as references.
- ☐ Labeled benchmark: bad PRNGs, good PRNGs, TRNG sources; report ROC/AUC for the
  predictability axis (pseudo vs true).
- ☐ Address/de-bias real quantum-hardware bias (von Neumann extractor) or document it.

## Workstream E — Tests, packaging, reproducibility

- ☐ Unit tests for every statistical test + the scoring pipeline.
- ☐ Null-calibration tests (FPR ≈ α; per-test p-values ≈ Uniform(0,1)).
- ☐ Scale/length-invariance regression tests.
- ☐ CI (`.github/workflows`), declare sklearn dep, version bump, CHANGELOG.
- ☐ Fix README code/signature mismatches (`pseudo.mersenne_twister(n=)` →
  `prandom(...)`; `quantum.qrng` → `qrandom`; `inspectRandom` prints, returns None).

## Workstream F — Regenerate analysis + paper

- ☐ Re-run `analysis/` (scoring, PCA, summaries) with corrected tests; regenerate plots.
- ☐ `paper/` skeleton: methods, calibration, validation, honest limitations.

## Notes / decisions log

- 2026-06-23: Plan approved. Publication target = methodology + software paper.
  Framing = quality scorer + separate predictability axis. Scoring = principled
  redesign. Validation = full (NIST STS / Dieharder / TestU01).
