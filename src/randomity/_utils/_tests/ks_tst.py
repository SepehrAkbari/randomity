import numpy as np

def ks_test(data, num_sims=2000, seed=0):
    """
    Discrete Kolmogorov-Smirnov goodness-of-fit against the uniform distribution.

    The classic continuous KS test (scipy.stats.kstest(..., 'uniform')) is INVALID for
    discrete integer data: it assumes a continuous reference, and rescaling to [0,1]
    via min/max pins the empirical CDF to 0 and 1 at the extremes, producing
    meaningless p-values. Here we compute the KS statistic
        D = max_s | F_emp(s) - F_uniform(s) |
    at the k support points and calibrate its null distribution by Monte-Carlo
    simulation of uniform i.i.d. samples of the same length (seeded for
    reproducibility).
    """
    data = np.asarray(data)
    min_val, max_val = int(np.min(data)), int(np.max(data))
    k = max_val - min_val + 1
    n = len(data)

    if k < 2 or n < 2:
        return {'ks_p': np.nan, 'ks_D': np.nan}

    # support points and the theoretical (discrete uniform) CDF at each
    support = np.arange(min_val, max_val + 1)
    theoretical_cdf = np.arange(1, k + 1) / k

    def ks_statistic(sample):
        counts = np.array([np.sum(sample <= s) for s in support], dtype=float)
        emp_cdf = counts / len(sample)
        return np.max(np.abs(emp_cdf - theoretical_cdf))

    observed_D = ks_statistic(data)

    rng = np.random.default_rng(seed)
    sim_D = np.empty(num_sims)
    for i in range(num_sims):
        sample = rng.integers(min_val, max_val + 1, size=n)
        sim_D[i] = ks_statistic(sample)

    p_value = (np.sum(sim_D >= observed_D) + 1) / (num_sims + 1)

    return {
        'ks_p': float(p_value),
        'ks_D': float(observed_D)
    }
