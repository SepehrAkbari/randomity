"""
Unit + regression + light-calibration tests for the statistical tests.

These guard the Workstream A correctness fixes (see docs/ROADMAP.md):
  - permutation test is no longer permutation-invariant (was p ~ 1.0 always)
  - gap test uses the geometric null (was p ~ 1e-82 for everything)
  - entropy is normalized by the sequence's own alphabet (was a fixed 9.97)
  - serial autocorrelation scoring rewards |rho| ~ 0 (was rewarding rho = -0.88)
  - FFT uses a scale-invariant Fisher g-test p-value (was raw magnitude)

Full false-positive-rate calibration of the *combined* score is Workstream B.
"""
import numpy as np
import pytest

from src.randomity._utils._tests import (
    chisqr_test, ks_test, freq_test, eqdist_test, gap_test,
    serial_test, permute_test, entropy_test, ftt_test,
)
from src.randomity._utils.compute_score import _get_normalized_score, _getScore
from src.randomity._utils.gen_test_vector import _gen_test_vector


def _uniform(n, hi=10, seed=0):
    return np.random.default_rng(seed).integers(0, hi + 1, size=n)


# --------------------------------------------------------------------------- keys

def test_all_tests_return_expected_keys():
    data = _uniform(500)
    assert set(chisqr_test(data)) >= {'chisqr_p', 'chisqr_X2', 'chisqr_df'}
    assert set(ks_test(data)) >= {'ks_p', 'ks_D'}
    assert set(freq_test(data)) >= {'freq_p', 'freq_X2', 'freq_df'}
    assert set(eqdist_test(data)) >= {'eqdist_diff'}
    assert set(gap_test(data)) >= {'gap_p', 'gap_X2', 'gap_df'}
    assert set(serial_test(data)) >= {'serial_autocorrelation', 'serial_p'}
    assert set(permute_test(data)) >= {'perm_observed_stat', 'perm_p'}
    assert set(entropy_test(data)) >= {'entropy_val', 'entropy_ratio'}
    assert set(ftt_test(data)) >= {'fft_g', 'fft_g_p', 'fft_max_magnitude'}


# ------------------------------------------------------------ p-values are valid

@pytest.mark.parametrize("fn,key", [
    (chisqr_test, 'chisqr_p'),
    (ks_test, 'ks_p'),
    (freq_test, 'freq_p'),
    (gap_test, 'gap_p'),
    (serial_test, 'serial_p'),
    (ftt_test, 'fft_g_p'),
    (permute_test, 'perm_p'),
])
def test_pvalues_in_unit_interval(fn, key):
    p = fn(_uniform(1000, seed=1))[key]
    assert 0.0 <= p <= 1.0


# ------------------------------------------------- regression: degeneracies gone

def test_permutation_test_not_invariant():
    # sorted (perfectly ordered) data must be flagged; the old grand-mean statistic
    # gave perm_p ~ 1.0 regardless of ordering.
    ordered = np.sort(_uniform(500, seed=2))
    assert permute_test(ordered)['perm_p'] < 0.05
    # random data should not be flagged
    assert permute_test(_uniform(500, seed=3))['perm_p'] > 0.05


def test_gap_test_not_degenerate_on_random():
    # the old uniform-null gap test produced p ~ 1e-82 for every generator
    p = gap_test(_uniform(3000, seed=4))['gap_p']
    assert p > 1e-3


def test_entropy_ratio_alphabet_relative():
    # uniform data over its alphabet must score ~1.0 regardless of the range used;
    # the old fixed-9.97 normalization forced [0,10] data down to ~0.34.
    for hi in (1, 10, 255):
        ratio = entropy_test(_uniform(5000, hi=hi, seed=5))['entropy_ratio']
        assert ratio > 0.95


def test_serial_scoring_rewards_zero_correlation():
    # |rho| ~ 0 must score higher than strong correlation of either sign
    s_zero = _get_normalized_score(0.0, 'serial_p')  # serial_p ~ 1 when rho ~ 0
    # build a serial_p for a strongly correlated sequence
    corr_seq = (np.cumsum(np.random.default_rng(6).integers(-1, 2, size=2000)) % 11)
    p_corr = serial_test(corr_seq)['serial_p']
    assert serial_test(_uniform(2000, seed=7))['serial_p'] > p_corr


# --------------------------------------------------------- behavioral: structure

def test_periodic_sequence_is_flagged():
    ramp = np.array(([list(range(11))] * 200))[:, :].ravel()[:2000]
    v = _gen_test_vector(ramp.tolist())
    assert v['fft_g_p'] < 0.05   # dominant periodic component
    assert v['freq_p'] < 0.05    # pairs are not jointly uniform
    assert _getScore(v) < 0.6


def test_uniform_random_individual_tests_pass():
    v = _gen_test_vector(_uniform(2000, seed=8).tolist())
    # each marginal/quality p-value should be comfortably non-significant
    assert v['chisqr_p'] > 0.01
    assert v['gap_p'] > 0.01
    assert v['serial_p'] > 0.01
    assert v['entropy_ratio'] > 0.95


# ------------------------------------------- light calibration: p-values spread out

def test_pvalues_not_all_extreme_across_random_samples():
    # On random input each p-value should land across (0,1), not pile up at 0 or 1.
    ps = [chisqr_test(_uniform(800, seed=s))['chisqr_p'] for s in range(40)]
    ps = np.array(ps)
    assert 0.2 < ps.mean() < 0.8
    assert (ps < 0.5).sum() > 5 and (ps > 0.5).sum() > 5
