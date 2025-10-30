"""
Hungarian Algorithm (Kuhn-Munkres) cho bài toán gán tổng chi phí tối thiểu.
API tương thích: hungarian_algorithm(cost_matrix) -> (min_cost, assignment)
Assignment: list[(row_idx, col_idx)]
"""

from math import inf, isfinite
from typing import List, Tuple

def hungarian_algorithm(cost_matrix: List[List[float]]) -> Tuple[float, List[Tuple[int, int]]]:
    n = len(cost_matrix)
    if n == 0:
        return 0, []
    m = len(cost_matrix[0])
    assert n == m, 'Ma trận chi phí phải vuông.'

    cost = [row[:] for row in cost_matrix]
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # tìm đường tăng luồng (augmenting path)
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    # Tạo gán (assignment)
    assignment = []
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment.append((p[j] - 1, j - 1))
    total_cost = sum(cost_matrix[i][j] for i, j in assignment)
    return total_cost, assignment

def minimum_matching_cost(cost_matrix: List[List[float]]) -> float:
    cost, _ = hungarian_algorithm(cost_matrix)
    return cost


