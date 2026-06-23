import numpy as np
from typing import Dict, Union
from .normalization_config import NORMALIZATION_PARAMS

# Features per paradigm. After the Workstream A correctness fixes, the inputs are:
#   - calibrated p-values: chisqr_p, ks_p, freq_p, gap_p, serial_p, fft_g_p
#       (higher p => more consistent with randomness => higher score, used directly)
#   - entropy_ratio in [0, 1]: 1.0 == maximally uniform (used directly)
#   - eqdist_diff: a magnitude in [0, 0.5], lower == more random (min-max + invert)
_FEATURES = {
    'uniformity': ['chisqr_p', 'ks_p', 'freq_p', 'eqdist_diff'],
    'patterns': ['gap_p', 'serial_p'],
    'periodicity': ['entropy_ratio', 'fft_g_p']
}

# Quantities already on a [0, 1] "more random == higher" scale (p-values + entropy ratio).
_DIRECT_SCORES = {'chisqr_p', 'ks_p', 'freq_p', 'gap_p', 'serial_p', 'fft_g_p', 'entropy_ratio'}


def _get_normalized_score(value: float, feature_name: str) -> float:
    """
    Normalizes a single test result value to a 0-1 score (higher == more random).

    NOTE: aggregation across tests is still a simple per-paradigm mean here. Replacing
    that with a principled combination of p-values (Fisher's method + empirical
    calibration for test dependence) is Workstream B in docs/ROADMAP.md.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    if feature_name in _DIRECT_SCORES:
        score = float(value)

    elif feature_name in NORMALIZATION_PARAMS:
        min_val = NORMALIZATION_PARAMS[feature_name]['min']
        max_val = NORMALIZATION_PARAMS[feature_name]['max']

        if min_val == max_val:
            score = 0.5
        else:
            normalized_value = (float(value) - min_val) / (max_val - min_val)
            # eqdist_diff is a "badness" magnitude: smaller deviation == more random
            if feature_name == 'eqdist_diff':
                score = 1.0 - normalized_value
            else:
                score = normalized_value
    else:
        score = 0.0

    return float(np.clip(score, 0.0, 1.0))


def _paradigm_score(test_results: Dict[str, Union[float, int]], paradigm: str) -> float:
    scores = [_get_normalized_score(test_results.get(feat, np.nan), feat)
              for feat in _FEATURES[paradigm]]
    valid = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan


def _howUniform(test_results: Dict[str, Union[float, int]]) -> float:
    """Computes the uniformity score from test results."""
    return _paradigm_score(test_results, 'uniformity')


def _howPatterns(test_results: Dict[str, Union[float, int]]) -> float:
    """Computes the patterns score from test results."""
    return _paradigm_score(test_results, 'patterns')


def _howPeriodic(test_results: Dict[str, Union[float, int]]) -> float:
    """Computes the periodicity score from test results."""
    return _paradigm_score(test_results, 'periodicity')


def _getScore(test_results: Dict[str, Union[float, int]]) -> float:
    """Computes the overall randomness score by averaging the sub-scores."""
    all_scores = [
        _howUniform(test_results),
        _howPatterns(test_results),
        _howPeriodic(test_results),
    ]
    valid = [s for s in all_scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan
