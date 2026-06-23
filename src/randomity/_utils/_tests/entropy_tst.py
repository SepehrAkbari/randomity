import numpy as np

def entropy_test(data):
    """
    Shannon entropy of the symbol distribution, normalized by the sequence's own
    alphabet size.

    H = -sum_j p_j log2(p_j) over the k = (max - min + 1) symbols. The maximum possible
    entropy is log2(k), achieved by the uniform distribution, so we report the
    scale-invariant ratio
        entropy_ratio = H / log2(k)  in [0, 1]
    where 1.0 means maximally uniform. The raw entropy (entropy_val) is kept for
    backward compatibility / the analysis pipeline.

    NOTE: the previous scoring normalized entropy by a fixed constant (9.97 =
    log2(~1000)), which ignored the alphabet size: a perfectly uniform sequence over
    [0, 10] (max entropy log2(11) ~= 3.46) was forced to ~0.34 and looked non-random.
    """
    data = np.asarray(data)
    min_val, max_val = int(np.min(data)), int(np.max(data))
    k = max_val - min_val + 1

    bins = np.arange(min_val, max_val + 2)  # one bin per integer symbol
    hist, _ = np.histogram(data, bins=bins)
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]

    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    max_entropy = np.log2(k) if k > 1 else np.nan
    entropy_ratio = float(entropy_value / max_entropy) if k > 1 else np.nan

    return {
        'entropy_val': float(entropy_value),
        'entropy_ratio': entropy_ratio
    }
