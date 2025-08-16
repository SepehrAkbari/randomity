import numpy as np
import os
import csv
from typing import List
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', ))
sys.path.insert(0, project_root)

from src.randomity.generate import (
    qrandom, 
    lcg, 
    middle_square, 
    mersenne_twister, 
    xor_shift, 
    blum_blum_shub
)

os.makedirs('data', exist_ok=True)

def save_sequence(sequence: List[int], filename: str):
    """Saves a sequence to a CSV file."""
    filepath = os.path.join('data', filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n'])
        for val in sequence:
            writer.writerow([val])

def gen_test_set(num_sequences: int = 100, sequence_size: int = 1000):
    print("Generating sequences for the test set...")
    
    # Source 1: NumPy's default
    print("Numpy is generating...")
    for i in range(num_sequences):
        seq = np.random.randint(0, 1000, size=sequence_size).tolist()
        filename = f"numpy_random_{i}.csv"
        save_sequence(seq, filename)
        
    # Source 2: LCG
    print("LCG is generating...")
    for i in range(num_sequences):
        seq = lcg(max_val=1000, num_out=sequence_size, seed=1234+i)
        seq_norm = [val % 1000 for val in seq]
        filename = f"lcg_good_{i}.csv"
        save_sequence(seq_norm, filename)

    # Source 2: LCG (bad parameters)
    print("LCG (with bad params) is generating...")
    for i in range(num_sequences):
        seq = lcg(max_val=1000, num_out=sequence_size, seed=1234+i, a=3, c=4, m=10)
        seq_norm = [val % 1000 for val in seq]
        filename = f"lcg_bad_{i}.csv"
        save_sequence(seq_norm, filename)
        
    # Source 3: Middle-Square Method
    print("Middle-Square is generating...")
    for i in range(num_sequences):
        seq = middle_square(max_val=1000, num_out=sequence_size, seed=1234 + i)
        seq_norm = [val % 1000 for val in seq]
        filename = f"midsquare_{i}.csv"
        save_sequence(seq_norm, filename)
        
    # Source 4: Highly Predictable Sequence
    print("Predicatable is generating...")
    for i in range(num_sequences):
        seq = [j % 1000 for j in range(sequence_size)]
        filename = f"predictable_{i}.csv"
        save_sequence(seq, filename)

    # Source 5: Quantum Randomness
    # print("QRNG is generating...")
    # gates = ["h", "rx", "ry", "sx"]
    # for i in range(num_sequences):
    #     seq = qrandom(max_val=1000, num_out=sequence_size, q_gate=gates[i % len(gates)])
    #     filename = f"qrandom_{i}.csv"
    #     save_sequence(seq, filename)

    # Source 6: Mersenne Twister
    print("Mersenne Twister is generating...")
    for i in range(num_sequences):
        seq = mersenne_twister(max_val=1000, num_out=sequence_size, seed=1234+i)
        filename = f"mt_{i}.csv"
        save_sequence(seq, filename)

    # source 7: XOR Shift
    print("XOR Shift is generating...")
    for i in range(num_sequences):
        seq = xor_shift(max_val=1000, num_out=sequence_size, seed=1234+i)
        filename = f"xorshift_{i}.csv"
        save_sequence(seq, filename)

    # source 8: Blum Blum Shub
    print("Blum Blum Shub is generating...")
    for i in range(num_sequences):
        seq = blum_blum_shub(max_val=1000, num_out=sequence_size)
        filename = f"bbs_{i}.csv"
        save_sequence(seq, filename)

    # Source 9: Incremental sequence
    print("Incremental is generating...")
    for i in range(num_sequences//2):
        seq = [j + 1 for j in range(sequence_size)]
        filename = f"incremental_{i}.csv"
        save_sequence(seq, filename)
    for i in range(num_sequences//2):
        seq = [j + 5 for j in range(sequence_size)]
        filename = f"incremental_{i}.csv"
        save_sequence(seq, filename)

    print("Test set generation complete.")

if __name__ == '__main__':
    gen_test_set()