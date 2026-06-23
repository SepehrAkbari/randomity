# Methodology

Single source of truth for the mathematics of `randomity`. The code in
`src/randomity/_utils/_tests/` and `_utils/compute_score.py` must match this document;
the paper draws from it. When you change a test's math, change it here first.

Notation: a sequence is `x = (x_1, ..., x_N)`, integers over an observed alphabet of
size `k = max(x) - min(x) + 1`. The null hypothesis H₀ throughout is "the `x_i` are
i.i.d. uniform over the `k` symbols," unless a test states otherwise. Every test is
required to emit a **p-value** calibrated under H₀.

---

## Axis 1 — Distributional quality

### Uniformity

**Chi-square frequency test.** Bin counts `O_j` over the `k` symbols vs expected
`E_j = N/k`. Statistic `X² = Σ (O_j − E_j)²/E_j ~ χ²_{k−1}` under H₀. Requires
`E_j ≳ 5`; collapse sparse bins otherwise. (Note: the legacy `chisqr_test` and
`freq_test` are the same test — keep one; the second slot should become a higher-order
uniformity test, e.g. on consecutive pairs / the serial test of order 2.)

**Equidistribution.** Deviation of the sample mean from `0.5` after scaling to `[0,1]`.
This is a weak corollary of the chi-square and currently has no calibrated p-value; in
the redesign it is either dropped or replaced by a proper test of the mean
(`z = (x̄ − 0.5)/(σ₀/√N)`, `σ₀² = (k²−1)/(12 k²)` for the discrete uniform on `k`
symbols scaled to `[0,1]`).

**Kolmogorov–Smirnov (discrete).** The classic `kstest(·, 'uniform')` is **invalid**
for discrete data (it assumes a continuous reference, and min/max pinning forces the
empirical CDF to 0 and 1 at the extremes). Replace with a discrete goodness-of-fit
(chi-square already covers this) or a discretized Cramér–von Mises statistic with a
Monte-Carlo null. If retained at all, document the discreteness caveat.

### Patterns / short-range structure

**Serial autocorrelation.** Lag-1 Pearson correlation `ρ = corr(x_{1:N−1}, x_{2:N})`.
Under H₀, for large `N`, `ρ √N → N(0,1)`, giving a two-sided p-value
`p = 2(1 − Φ(√N |ρ|))`. The *score* must reward `|ρ| ≈ 0` (both signs of correlation
are non-random) — the legacy min-max+invert scoring incorrectly rewarded `ρ = −0.88`.

**Gap test.** For a chosen symbol (or bin), the gaps between successive occurrences are
**geometrically** distributed under H₀: `P(gap = g) = p(1−p)^{g−1}`, with `p` the
symbol's probability (`1/k` for a single symbol, or the bin width). Test observed gap
frequencies against this geometric pmf via chi-square (pooling the tail so each
expected count `≳ 5`); df = (#gap classes − 1). The legacy test compared gaps to a
*uniform* expectation, which is wrong and made every generator fail.

**Runs / permutation.** The legacy permutation test used `mean(block_means)`, which
equals the grand mean and is **invariant under permutation** (p ≈ 1.0 always — measures
nothing). Replace with either: (a) the **runs test** (count runs above/below the
median; under H₀ the run count has a known mean/variance ⇒ normal approximation), or
(b) a permutation test on a statistic that *is* sensitive to ordering, e.g. the
variance of block means or the sum of squared lag-1 differences.

### Periodicity / spectrum

**Spectral (FFT) test.** Compute the periodogram `I_f = |X_f|² / N` of the
mean-centered sequence (drop the DC term). Under H₀ the normalized peaks follow a known
distribution; **Fisher's g-test** uses `g = max_f I_f / Σ_f I_f` with an exact null
distribution, giving a calibrated p-value for "is there a dominant periodic component?"
Raw FFT magnitude (legacy) is **not** scale- or length-invariant and must not be used
directly. (Optionally mirror the NIST STS spectral test: fraction of peaks below the
95% threshold.)

**Shannon entropy.** `H = −Σ_j p_j log₂ p_j` over the `k` symbols. Report the
**entropy ratio** `H / log₂ k ∈ [0,1]` — normalized by the *sequence's own* alphabet,
not a fixed constant. For a calibrated p-value, use the asymptotic distribution of the
plug-in entropy estimator, or a Monte-Carlo null. (Better still, use the **entropy
rate** / block entropy, which connects to the predictability axis below.)

---

## Axis 2 — Predictability (the PRNG-vs-TRNG signal)

These measure how well the next symbol can be predicted from the past — the property
that genuinely separates pseudo from true randomness. Reported **separately** from the
quality axis.

- **Compression ratio.** Compress the byte/bit representation (e.g. zlib/bz2). True
  randomness is incompressible (ratio ≈ 1); exploitable structure compresses. Provides
  an upper bound on entropy rate.
- **Maurer's universal test** (NIST STS): estimates the per-bit entropy by mean log
  gap between recurrences of L-bit blocks; has a known null mean/variance ⇒ p-value.
- **Linear complexity / Berlekamp–Massey** (NIST STS): length of the shortest LFSR
  reproducing the bit stream. Short complexity ⇒ predictable. Block-wise statistic has
  a known χ² null.
- **Learned next-symbol predictor.** Train a small model (n-gram / logistic / shallow
  MLP) on `(window → next symbol)`; test-set accuracy significantly above `1/k` ⇒
  predictable. This is the novel, defensible distinguisher. Report accuracy with a
  binomial-test p-value vs the chance rate.

---

## Aggregation

1. Each test → a p-value `p_i` under H₀.
2. Combine within each paradigm and overall with **Fisher's method**:
   `T = −2 Σ_i ln p_i`. Under independence `T ~ χ²_{2m}`. Because the tests are
   **not** independent, calibrate `T`'s null **empirically** by Monte-Carlo over many
   genuinely-uniform sequences (matched in length/alphabet), or restrict to a
   near-independent subset. Report the empirical combined p-value.
3. The **decision** (`isRandom`) rejects H₀ at level α when the combined p-value < α.
   Report the realized false-positive rate on uniform input (must be ≈ α).
4. The predictability axis is reported and thresholded **separately** — it answers a
   different question and must not be averaged into the quality score.

### Calibration requirements (tested in CI)

- Feed N(≥1000) genuinely-uniform sequences; the fraction with combined p < α must be
  ≈ α (within Monte-Carlo error).
- Each individual test's p-value must be ≈ Uniform(0,1) on uniform input
  (KS test of the p-value distribution).
- Scale/length invariance: identical verdict distribution across ranges
  `{[0,1],[0,10],[0,255]}` and lengths `{500, 5000, 50000}` (up to power).
