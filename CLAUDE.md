# CLAUDE.md

Context file for AI assistants (and humans) working on `randomity`. Keep this file
current — it is the persistent memory of the project's purpose, architecture, and the
mathematical conventions the code must obey.

## What this project is

`randomity` is an open-source Python package plus a research/analysis repository for
**assessing the randomness of integer sequences** and **generating** them (classical
PRNGs + a Qiskit-based QRNG). It is being prepared for a **methodology + software
publication** (target: *Journal of Statistical Software* / *SoftwareX* style).

## The central scientific framing (read this before changing any test)

Randomness is assessed along **two distinct, non-interchangeable axes**:

1. **Distributional quality** — does the sequence *look* like i.i.d. uniform draws?
   (uniformity, absence of short-range patterns, flat spectrum). This is what the
   original nine tests measure.
2. **Predictability** — can the next symbol be guessed better than chance given the
   past? (compression, linear complexity, learned predictors). This is what truly
   separates a PRNG from a TRNG.

**Critical caveat that drives the whole design:** statistical *quality* tests cannot
distinguish a good PRNG (e.g. Mersenne Twister) from a TRNG — good PRNGs are
*designed* to pass them. Only the **predictability** axis can. Never claim the quality
score detects pseudo-vs-true randomness; that claim belongs to the predictability axis
alone. The package reports the two axes **separately**.

## Mathematical conventions (every test must follow these)

- **Every test returns a calibrated p-value** under the null H₀ = "i.i.d. uniform over
  the sequence's observed alphabet." Magnitudes (entropy, |ρ|, spectral peak) are
  converted to p-values via their null distribution or a Monte-Carlo reference.
- **Combine p-values with Fisher's method** (−2·Σ ln pᵢ ~ χ²₂ₖ) or Stouffer's Z — never
  by averaging p-values (averaging has no distributional meaning).
- **Scale and length invariance**: a test's verdict must not depend on the sequence's
  numeric range or (beyond power) its length. Entropy is normalized by log₂(alphabet
  size); spectral magnitude is normalized by total power; correlations are unit-free.
- **Multiple-testing control**: report per-test p-values with a documented α (or
  Benjamini–Hochberg FDR). The suite's empirical false-positive rate on genuinely
  random input must be ≈ α — this is a required calibration test.

## Architecture

```
src/randomity/            # the installable package
  generate/               # PRNG + QRNG generation
    pseudo.py             # prandom(...) unified PRNG API + per-algo functions
    quantum.py            # qrandom(...) Qiskit QRNG
    algos/                # MT19937, XORShift, LCG, BlumBlumShub, MiddleSquare, MTNumpy
  evaluate/               # public assessment API
    isRandom.py           # bool decision vs threshold
    inspectRandom.py      # per-paradigm score report
    whyRandom.py          # per-test contribution report (STUB — to implement)
  _utils/
    _tests/               # the statistical tests (one file each)
    gen_test_vector.py    # runs all tests -> result dict
    compute_score.py      # normalization + aggregation -> composite score
    normalization_config.py  # min/max params (being phased out, see below)
analysis/                 # research pipeline (NOT shipped)
  data-gen/               # scripts that generated the PoC datasets
  data/                   # PoC sequences (LCG/MT/XORShift/Q) + feature vectors
  normalization-training/ # how normalization params were fit
  pca/                    # PCA of the feature vectors
  scoring/                # composite scoring of the analysis data
  test-suite/             # R reference implementations of the tests
tests/                    # pytest suite (currently generate/ only)
docs/                     # METHODOLOGY.md (math spec), ROADMAP.md (workstreams)
```

`analysis/` mirrors the package's scoring logic; when the package's tests/scoring
change, the analysis must be regenerated (Workstream F) to stay consistent.

## Build / test / dev commands

```bash
pip install -e .                       # editable install
pip install -r requirements.txt        # qiskit, numpy, scipy, matplotlib (+ sklearn for PCA)
pytest                                  # run tests (imports use `from src.randomity...`)
python -m build                        # build wheel/sdist (hatchling backend)
```

Tests import as `from src.randomity...` and are run from the repo root.

## Known issues / live work (keep in sync with docs/ROADMAP.md)

The scoring methodology is under a principled redesign. Confirmed correctness bugs in
the *original* implementation (status tracked in ROADMAP):

- **Permutation test** (`permute_tst.py`): statistic = grand mean, invariant under
  permutation ⇒ p ≈ 1.0 always. Test measured nothing.
- **Gap test** (`gap_tst.py`): tested recurrence gaps against a *uniform* null; gaps are
  **geometric** ⇒ every generator failed at p ≈ 1e-82.
- **KS test** (`ks_tst.py`): `kstest(...,'uniform')` is invalid on discrete integer data;
  min/max pinning distorts the CDF ⇒ meaningless p-values.
- **Entropy normalization**: fixed max 9.97 ignores the sequence's alphabet size ⇒
  perfectly random [0,10] data scored ≈0.34. Must normalize by log₂(alphabet).
- **Serial autocorrelation scoring** (`compute_score.py`): rewards ρ = −0.88 as "most
  random"; should reward |ρ| ≈ 0.
- **FFT magnitude**: not scale/length invariant; fixed normalization max is
  dataset-specific.
- **Aggregation**: averaged p-values (no statistical meaning) — replace with Fisher's.
- **Redundancy**: `chisqr_test` and `freq_test` are near-identical.

## Conventions for contributors / assistants

- Keep the public API (`isRandom`, `inspectRandom`, `prandom`, `qrandom`) stable; change
  internals freely.
- Match the existing code style (numpy-based, small single-purpose test modules).
- Any new statistical test: return a p-value, add a docstring stating its null
  hypothesis, and add both a behavioral test and a null-calibration test under `tests/`.
- Update `docs/METHODOLOGY.md` whenever a test's math changes — it is the single source
  of truth shared by the code and the paper.
