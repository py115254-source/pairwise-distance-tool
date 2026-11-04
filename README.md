
# Pairwise Euclidean Distance Tool

A Python command-line utility that computes pairwise Euclidean distances between two sets of points using NumPy broadcasting.

## Features
- Generate random 2-D point sets using NumPy
- Compute distance matrices efficiently with broadcasting
- Reusable `pairwise_distances()` function
- Validate results against SciPy's `cdist`
- Load point coordinates from a CSV file
- Command-line interface with input validation

## Requirements
- Python 3.10+
- NumPy
- SciPy

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Generate random point sets:
```bash
python main.py
```

Generate 10 points in each set:
```bash
python main.py --points 10
```

Load points from a CSV file:
```bash
python main.py --file points.csv
```

The CSV file must contain four columns: `ax, ay, bx, by`.

## Validation
The utility compares its distance matrix with SciPy's `cdist` and reports whether the results match.

## Disclaimer
This project is for educational purposes.
