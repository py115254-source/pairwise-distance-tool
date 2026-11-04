
"""
Pairwise Euclidean distance utility using NumPy broadcasting.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between every pair of points."""
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)

    if points_a.ndim != 2 or points_b.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays.")

    if points_a.shape[1] != points_b.shape[1]:
        raise ValueError("Both point sets must have the same dimensions.")

    differences = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=2))


def generate_random_points(
    num_points: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two random sets of 2-D points."""
    rng = np.random.default_rng(seed)
    points_a = rng.random((num_points, 2))
    points_b = rng.random((num_points, 2))
    return points_a, points_b


def validate_with_scipy(
    points_a: np.ndarray, points_b: np.ndarray
) -> bool:
    """Compare the custom distance matrix with SciPy's cdist."""
    custom_result = pairwise_distances(points_a, points_b)
    scipy_result = cdist(points_a, points_b, metric="euclidean")
    return np.allclose(custom_result, scipy_result)


def main() -> int:
    """Load points from a CSV file or generate random points."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise Euclidean distances between points."
    )
    parser.add_argument(
        "--file",
        help="CSV file containing x,y coordinates (two columns)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=5,
        help="Number of random points in each set (default: 5)",
    )
    args = parser.parse_args()

    if args.points <= 0:
        parser.error("--points must be a positive integer")

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            parser.error(f"File not found: {file_path}")

        try:
            points = np.loadtxt(file_path, delimiter=",", ndmin=2)
        except ValueError as error:
            parser.error(f"Invalid CSV point data: {error}")

        if points.shape[1] != 4:
            parser.error("CSV must contain four columns: ax,ay,bx,by")

        points_a = points[:, :2]
        points_b = points[:, 2:]
    else:
        points_a, points_b = generate_random_points(args.points)

    distances = pairwise_distances(points_a, points_b)

    print("Points A:")
    print(points_a)
    print("\nPoints B:")
    print(points_b)
    print("\nPairwise Euclidean distance matrix:")
    print(distances)
    print("\nSciPy validation:", validate_with_scipy(points_a, points_b))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
