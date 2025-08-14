import os
import re

import numpy as np

from ..evaluate.tests.chisqr_tst import chisqr_test
from ..evaluate.tests.ks_tst import ks_test
from ..evaluate.tests.freq_tst import freq_test
from ..evaluate.tests.eqdist_tst import eqdist_test
from ..evaluate.tests.gap_tst import gap_test
from ..evaluate.tests.serial_tst import serial_test
from ..evaluate.tests.permute_tst import permute_test
from ..evaluate.tests.entropy_tst import entropy_test
from ..evaluate.tests.ftt_tst import ftt_test

def gen_test_vector(sequence: list) -> dict:
    """
    Generate a result vector from the given sequence of numbers.

    Args:
        sequence (list): A list of integers representing the sequence to be tested.

    Returns:
        dict: A dictionary containing the results of various statistical tests.
    """
    seq_numbers = np.array(sequence)

    test_results = {}
    test_results.update(chisqr_test(seq_numbers))
    test_results.update(ks_test(seq_numbers))
    test_results.update(freq_test(seq_numbers))
    test_results.update(eqdist_test(seq_numbers))
    test_results.update(gap_test(seq_numbers))
    test_results.update(serial_test(seq_numbers))
    test_results.update(permute_test(seq_numbers))
    test_results.update(entropy_test(seq_numbers))
    test_results.update(ftt_test(seq_numbers))

    return test_results
