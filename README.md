# randomity

`randomity` is a comprehensive statistical suite and Python library designed to measure the unpredictability of pseudo-random number generators (PRNGs) and true random number generators (TRNGs). The package provides tools to generate sequences using industry-standard algorithms, implement quantum-generated randomness via Qiskit, and evaluate those sequences across multiple statistical dimensions to compute a composite randomness score.

The project implements several generation strategies, including Linear Congruential Generators (LCG), Mersenne Twister, XORShift, Blum Blum Shub, and Middle Square algorithms, alongside a Quantum Random Number Generator (QRNG) that leverages single-qubit measurement collapse.

## Approach

The core of the package is a multi-dimensional evaluation engine that categorizes randomness into three primary paradigms: Uniformity, Patterns, and Periodicity. Each paradigm was evaluated using a suite of statistical tests. To provide a clear assessment, `randomity` implements a normalization pipeline that maps disparate statistical metrics (such as p-values and spectral magnitudes) onto a standardized $[0,1]$ scale. These normalized values are then synthesized into a composite Randomness Score, allowing for a direct, comparative evaluation of different generators.

## Usage

To install and use `randomity`:

```bash
pip install randomity
```

#### Evaluating a Sequence

You can use the `isRandom` function to determine if a sequence meets a specific unpredictability threshold (default is $0.6$):

```python
from randomity.evaluate import isRandom

# A sample sequence of numbers
my_sequence = [7, 1, 0, 4, 1, 9, 3, 2, 8, 5]

# Returns a boolean based on the composite score
result = isRandom(my_sequence, threshold=0.6)
print(f"Is it random? {result}")
```

#### Generating Random Sequences

The package provides dedicated modules for both pseudo and quantum generation:

```python
from randomity.generate import pseudo, quantum

# Generate a sequence using the Mersenne Twister algorithm
prng_seq = pseudo.mersenne_twister(n=1000)

# Generate a sequence using a Quantum Random Number Generator (requires qiskit)
qrng_seq = quantum.qrng(n=100)
```

### Inspection

For a deeper look into why a sequence passed or failed, you can inspect the individual scores for each paradigm:

```python
from randomity.evaluate import inspectRandom

# Get a dictionary of sub-scores for Uniformity, Patterns, and Periodicity
report = inspectRandom(my_sequence)
print(report)
```

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).