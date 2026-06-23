import numpy as np
from scipy.special import comb

def ftt_test(data):
    """
    Spectral test for periodicity using Fisher's g-test.

    The periodogram of a random sequence is flat in expectation; a periodic component
    shows up as a dominant peak. Fisher's g-statistic is the largest periodogram
    ordinate relative to the total power,
        g = max_f I_f / sum_f I_f,
    which is scale- and (largely) length-invariant. Its exact null distribution gives
    a p-value (large g, small p => significant periodicity => less random):
        P(g > x) = sum_{j>=1} (-1)^(j-1) C(m, j) (1 - j x)^(m-1),   0 < x < 1.

    NOTE: the previous test used the raw FFT magnitude, which grows with N and with the
    value range, so the fixed normalization constant (~1.5e7) was dataset-specific and
    did not generalize. fft_max_magnitude / dominant frequency are kept for backward
    compatibility but fft_g_p is the scale-invariant quantity used for scoring.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 4:
        return {'fft_dominant_frequency': np.nan, 'fft_dominant_period': np.nan,
                'fft_max_magnitude': np.nan, 'fft_g': np.nan, 'fft_g_p': np.nan}

    # mean-center so the DC term does not dominate
    centered = data - np.mean(data)
    fft_result = np.fft.rfft(centered)
    power = np.abs(fft_result) ** 2

    # drop DC (index 0) and Nyquist (last, if n even) to use the standard m ordinates
    periodogram = power[1:(n // 2)] if n % 2 == 0 else power[1:]
    m = len(periodogram)
    total_power = np.sum(periodogram)

    if m < 2 or total_power <= 0:
        return {'fft_dominant_frequency': np.nan, 'fft_dominant_period': np.nan,
                'fft_max_magnitude': np.nan, 'fft_g': np.nan, 'fft_g_p': np.nan}

    dominant_index = int(np.argmax(periodogram))
    dominant_magnitude = np.sqrt(periodogram[dominant_index])
    frequencies = np.fft.rfftfreq(n)[1:m + 1]
    dominant_frequency = frequencies[dominant_index]
    dominant_period = 1.0 / dominant_frequency if dominant_frequency > 0 else np.nan

    g = periodogram[dominant_index] / total_power

    # Fisher's exact p-value via inclusion-exclusion. Only j with (1 - j*g) > 0 contribute;
    # cap the number of terms for numerical safety (extra terms matter only when p ~ 1).
    p_value = 0.0
    j_max = int(np.floor(1.0 / g))
    for j in range(1, min(j_max, 30) + 1):
        term = comb(m, j, exact=True) * (1.0 - j * g) ** (m - 1)
        p_value += -term if j % 2 == 0 else term
    p_value = float(np.clip(p_value, 0.0, 1.0))

    return {
        'fft_dominant_frequency': float(dominant_frequency),
        'fft_dominant_period': float(dominant_period) if np.isfinite(dominant_period) else np.nan,
        'fft_max_magnitude': float(dominant_magnitude),
        'fft_g': float(g),
        'fft_g_p': p_value
    }
