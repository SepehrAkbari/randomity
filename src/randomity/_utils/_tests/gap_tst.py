import numpy as np
from scipy.stats import chisquare

def gap_test(data):
    """
    Gap test against the correct (geometric) null.

    For a symbol that occurs with probability p under H0 (uniform i.i.d. over k
    symbols, p = 1/k), the gap g >= 1 between successive occurrences of that symbol is
    geometrically distributed:
        P(gap = g) = (1 - p)^(g-1) * p.
    We pool the gaps observed for every symbol and compare their frequencies to this
    geometric expectation with a chi-square test (tail cells pooled so every expected
    count >= 5).

    NOTE: the previous implementation compared gap frequencies to a *uniform*
    expectation, which is the wrong null; geometric gaps then looked extremely
    non-uniform and every generator failed at p ~ 1e-82.
    """
    data = np.asarray(data)
    min_val, max_val = int(np.min(data)), int(np.max(data))
    k = max_val - min_val + 1
    if k < 2:
        return {'gap_p': np.nan, 'gap_X2': np.nan, 'gap_df': np.nan}

    p = 1.0 / k

    gaps = []
    for value in range(min_val, max_val + 1):
        indices = np.where(data == value)[0]
        if len(indices) > 1:
            gaps.extend(np.diff(indices).tolist())

    G = len(gaps)
    if G < 5:
        return {'gap_p': np.nan, 'gap_X2': np.nan, 'gap_df': np.nan}

    gaps = np.asarray(gaps)
    max_gap = int(gaps.max())

    # observed counts for gap lengths 1..max_gap
    observed = np.bincount(gaps, minlength=max_gap + 1)[1:]

    # geometric expected counts; pool the right tail into a single cell so its
    # expected mass is the full remaining tail (keeps sum(expected) == G)
    obs_pooled, exp_pooled = [], []
    cur_obs = 0.0
    cur_exp = 0.0
    for g in range(1, max_gap + 1):
        cur_obs += observed[g - 1]
        cur_exp += G * (p * (1 - p) ** (g - 1))
        if cur_exp >= 5.0:
            obs_pooled.append(cur_obs)
            exp_pooled.append(cur_exp)
            cur_obs = 0.0
            cur_exp = 0.0
    # final tail cell: remaining observed + remaining geometric tail probability
    tail_obs = cur_obs
    tail_exp = G - sum(exp_pooled)
    if exp_pooled and tail_exp < 5.0:
        # merge the leftover tail into the last cell
        obs_pooled[-1] += tail_obs
        exp_pooled[-1] += tail_exp
    else:
        obs_pooled.append(tail_obs)
        exp_pooled.append(tail_exp)

    obs_pooled = np.asarray(obs_pooled, dtype=float)
    exp_pooled = np.asarray(exp_pooled, dtype=float)

    if len(obs_pooled) < 2 or not np.all(exp_pooled > 0):
        return {'gap_p': np.nan, 'gap_X2': np.nan, 'gap_df': np.nan}

    # rescale expected to match observed total exactly (guards float drift)
    exp_pooled *= obs_pooled.sum() / exp_pooled.sum()

    chi2_stat, p_value = chisquare(obs_pooled, f_exp=exp_pooled)

    return {
        'gap_p': float(p_value),
        'gap_X2': float(chi2_stat),
        'gap_df': len(obs_pooled) - 1
    }
