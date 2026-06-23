import numpy as np

def permute_test(data, num_permutations=1000, seed=0):
    """
    Permutation test for serial structure (ordering).

    Null hypothesis H0: the observed ordering is exchangeable (i.i.d.), i.e. the
    sequence carries no serial dependence.

    The statistic is the mean square successive difference (von Neumann):
        D = sum_i (x_{i+1} - x_i)^2
    Unlike the grand mean (which is invariant under permutation), D is sensitive to
    ordering: serially correlated data has systematically smaller (positive
    autocorrelation) or larger (negative autocorrelation) successive differences than
    a random reshuffle. We compare the observed D to its permutation distribution and
    return a two-sided p-value.

    NOTE: the previous implementation used mean(block_means), which equals the grand
    mean and is invariant under permutation, so its p-value was ~1.0 by construction.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 3 or np.var(data) == 0:
        return {'perm_observed_stat': np.nan, 'perm_p': np.nan}

    def successive_diff_stat(seq):
        return np.sum(np.diff(seq) ** 2)

    observed_stat = successive_diff_stat(data)

    rng = np.random.default_rng(seed)
    permuted_stats = np.empty(num_permutations)
    for i in range(num_permutations):
        permuted_stats[i] = successive_diff_stat(rng.permutation(data))

    center = np.mean(permuted_stats)
    # two-sided p-value with +1 correction (the observed value is itself a permutation)
    extreme = np.abs(permuted_stats - center) >= np.abs(observed_stat - center)
    p_value = (np.sum(extreme) + 1) / (num_permutations + 1)

    return {
        'perm_observed_stat': observed_stat,
        'perm_p': float(p_value)
    }
