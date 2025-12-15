import numpy as np
import random
from typing import List, Tuple


class TSP:
    """
    Input:
        points = [(x1, y1), (x2, y2), ...]
    Output:
        best_path: list of city indices
        best_length: total tour length (closed tour)
        0: placeholder
    """

    def __init__(self, points: List[Tuple[float, float]]):
        self.points = np.array(points, dtype=float)
        self.n = len(points)
        if self.n < 2:
            raise ValueError("points must contain at least 2 points.")
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Compute a symmetric pairwise Euclidean distance matrix."""
        n = self.n
        dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.points[i] - self.points[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _calculate_path_length(self, path: List[int]) -> float:
        """Compute tour length and close the loop (last city back to the first)."""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.distance_matrix[path[i]][path[i + 1]]
        total += self.distance_matrix[path[-1]][path[0]]
        return total

    def AR_ACO(
        self,
        n_ants: int = 50,
        n_iterations: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        seed: int | None = None,
    ):
        """
        AR_ACO (Adaptive Reconstruction ACO):

        Args:
            n_ants: number of ants per iteration
            n_iterations: number of ACO iterations
            alpha: pheromone importance
            beta: heuristic importance (1/distance)
            rho: evaporation rate
            seed: random seed for reproducibility

        Returns:
            (best_path, best_length, 0)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        pheromone = np.ones((self.n, self.n), dtype=float)
        best_path = None
        best_length = float("inf")

        def two_opt(path: List[int]) -> List[int]:
            improved = True
            best_len = self._calculate_path_length(path)
            while improved:
                improved = False
                for i in range(1, len(path) - 2):
                    for j in range(i + 1, len(path)):
                        if j - i == 1:
                            continue
                        new_path = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]
                        new_len = self._calculate_path_length(new_path)
                        if new_len < best_len:
                            path = new_path
                            best_len = new_len
                            improved = True
                            break
                    if improved:
                        break
            return path

        def three_opt(path: List[int]) -> List[int]:
            improved = True
            while improved:
                improved = False
                for i in range(self.n - 2):
                    for j in range(i + 1, self.n - 1):
                        for k in range(j + 1, self.n):
                            a, b = path[i], path[i + 1]
                            c, d = path[j], path[j + 1]
                            e, f = path[k], path[(k + 1) % self.n]

                            d0 = self.distance_matrix[a][b] + self.distance_matrix[c][d] + self.distance_matrix[e][f]
                            d1 = self.distance_matrix[a][c] + self.distance_matrix[b][d] + self.distance_matrix[e][f]
                            d2 = self.distance_matrix[a][b] + self.distance_matrix[c][e] + self.distance_matrix[d][f]
                            d3 = self.distance_matrix[a][d] + self.distance_matrix[e][b] + self.distance_matrix[c][f]

                            if d1 < d0:
                                path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
                                improved = True
                            elif d2 < d0:
                                path[j + 1 : k + 1] = reversed(path[j + 1 : k + 1])
                                improved = True
                            elif d3 < d0:
                                temp = path[i + 1 : k + 1]
                                path[i + 1 : k + 1] = temp[(j - i) :] + temp[: (j - i)]
                                improved = True

                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return path

        for _ in range(n_iterations):
            all_paths = []
            all_lengths = []

            for _ in range(n_ants):
                path = [0]
                unvisited = set(range(1, self.n))

                # Construct a tour
                while unvisited:
                    current = path[-1]
                    candidates = list(unvisited)

                    probs = []
                    for nxt in candidates:
                        pher = pheromone[current][nxt] ** alpha
                        dist = self.distance_matrix[current][nxt]
                        heuristic = (1.0 / (dist + 1e-12)) ** beta  # avoid division by zero
                        probs.append(pher * heuristic)

                    probs = np.array(probs, dtype=float)
                    probs /= probs.sum()
                    next_city = random.choices(candidates, weights=probs)[0]

                    path.append(next_city)
                    unvisited.remove(next_city)

                raw_length = self._calculate_path_length(path)
                mean_edge = np.mean([self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)])

                if raw_length / self.n < mean_edge * 1.2:
                    path = two_opt(path)
                else:
                    path = three_opt(path)

                length = self._calculate_path_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path[:]

            pheromone *= (1 - rho)
            for p, L in zip(all_paths, all_lengths):
                deposit = 1.0 / (L + 1e-12)
                for i in range(self.n):
                    a = p[i]
                    b = p[(i + 1) % self.n]
                    pheromone[a][b] += deposit
                    pheromone[b][a] += deposit  # keep symmetry

        return best_path, best_length, 0


if __name__ == "__main__":
    # Quick sanity test
    pts = [(random.random() * 100, random.random() * 100) for _ in range(30)]
    solver = TSP(pts)
    path, length, _ = solver.AR_ACO(n_ants=50, n_iterations=20, seed=0)
    print("best length:", length)
    print("path:", path)
