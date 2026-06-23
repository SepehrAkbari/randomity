import numpy as np
from scipy.stats import chisquare

def freq_test(data):
    """
    Serial (order-2) frequency test on consecutive non-overlapping pairs.

    The legacy freq_test was identical to chisqr_test (both: histogram the values, then
    chi-square vs uniform), double-counting the same evidence in the uniformity score.
    This version instead tests the joint uniformity of consecutive pairs
    (x_0, x_1), (x_2, x_3), ... over a b x b grid -- a genuine test of pairwise
    independence that the order-1 frequency test cannot see. The number of bins b per
    axis is chosen so the expected count per cell is ~5 (so the chi-square is valid even
    for short sequences); df = b^2 - 1.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 4:
        return {'freq_p': np.nan, 'freq_X2': np.nan, 'freq_df': np.nan}

    pairs = data[:2 * (n // 2)].reshape(-1, 2)
    n_pairs = len(pairs)
    min_val, max_val = float(np.min(data)), float(np.max(data))
    if max_val == min_val:
        return {'freq_p': np.nan, 'freq_X2': np.nan, 'freq_df': np.nan}

    k = int(round(max_val - min_val)) + 1
    # bins per axis: at most the alphabet size, sized so expected count per cell ~5
    b = max(2, min(k, int(np.sqrt(n_pairs / 5.0))))

    edges = np.linspace(min_val, max_val, b + 1)
    hist2d, _, _ = np.histogram2d(pairs[:, 0], pairs[:, 1], bins=[edges, edges])
    observed = hist2d.ravel()
    expected = np.full_like(observed, n_pairs / (b * b), dtype=float)

    if not np.all(expected > 0) or len(observed) < 2:
        return {'freq_p': np.nan, 'freq_X2': np.nan, 'freq_df': np.nan}

    chi2_stat, p_value = chisquare(observed, f_exp=expected)

    return {
        'freq_p': float(p_value),
        'freq_X2': float(chi2_stat),
        'freq_df': len(observed) - 1
    }
