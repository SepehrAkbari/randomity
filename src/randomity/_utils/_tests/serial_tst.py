import numpy as np
from scipy.stats import norm

def serial_test(data):
    """
    Lag-1 serial autocorrelation test.

    Under H0 (i.i.d.), the lag-1 Pearson autocorrelation rho satisfies
    sqrt(N) * rho -> N(0, 1) for large N, giving the two-sided p-value
        serial_p = 2 * (1 - Phi(sqrt(N) * |rho|)).
    Both signs of correlation indicate non-randomness, so the scoring rewards
    |rho| ~= 0 (equivalently a large serial_p).

    NOTE: the previous scoring min-max normalized rho over [-0.88, 1.0] and inverted
    it, which gave rho = -0.88 (strong negative correlation) the *best* score and
    rho = 0 only ~0.53. The directionally-correct quantity is |rho| (or serial_p).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 3 or np.var(data) == 0:
        return {'serial_autocorrelation': np.nan, 'serial_p': np.nan}

    rho = np.corrcoef(data[:-1], data[1:])[0, 1]

    z = np.sqrt(n) * abs(rho)
    p_value = 2.0 * (1.0 - norm.cdf(z))

    return {
        'serial_autocorrelation': float(rho),
        'serial_p': float(p_value)
    }
