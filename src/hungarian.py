"""
Giải thuật Hungarian cho bài toán gán
"""

import copy
from math import isfinite
from typing import List, Tuple


def hungarian_algorithm(cost_matrix: List[List[float]]) -> Tuple[float, List[Tuple[int, int]]]:
	"""
	Giải thuật Hungarian cho bài toán gán
	
	Args:
		cost_matrix: n x n matrix of costs
		
	Returns:
		tuple: (chi phí tối thiểu, danh sách các gán)
	"""
	n = len(cost_matrix)
	if n == 0:
		return 0, []
	
	# Tạo bản sao để tránh sửa đổi bản gốc
	matrix = [row[:] for row in cost_matrix]
	
	# Step 1: Trừ giá trị nhỏ nhất của hàng (bỏ qua NaN/Inf khi tính toán)
	for i in range(n):
		finite_values = [value for value in matrix[i] if isfinite(value)]
		row_min = min(finite_values) if finite_values else 0.0
		for j in range(n):
			if isfinite(matrix[i][j]) and row_min != 0.0:
				matrix[i][j] -= row_min
	
	
	# Step 2: Trừ giá trị nhỏ nhất của cột (bỏ qua NaN/Inf khi tính toán)
	for j in range(n):
		col_values = [matrix[i][j] for i in range(n) if isfinite(matrix[i][j])]
		col_min = min(col_values) if col_values else 0.0
		for i in range(n):
			if isfinite(matrix[i][j]) and col_min != 0.0:
				matrix[i][j] -= col_min
	
	# Step 3: Tìm gán tối đa
	assignment = _find_maximum_matching(matrix)
	# Tính tổng chi phí sử dụng ma trận gốc
	total_cost = 0
	for i, j in assignment:
		total_cost += cost_matrix[i][j]
	
	return total_cost, assignment


def _find_maximum_matching(matrix: List[List[float]]) -> List[Tuple[int, int]]:
	"""
    Tìm gán tối đa trong ma trận chi phí giảm
	"""
	n = len(matrix)
	assignment = []
	assigned_cols = [False] * n
	
	# Gán tham lam: gán mỗi hàng với cột tốt nhất có sẵn
	for i in range(n):
		best_col = -1
		best_cost = float('inf')
		
		for j in range(n):
			if not assigned_cols[j] and matrix[i][j] < best_cost:
				best_cost = matrix[i][j]
				best_col = j
		
		if best_col != -1:
			assigned_cols[best_col] = True
			assignment.append((i, best_col))
	return assignment


def minimum_matching_cost(cost_matrix: List[List[float]]) -> float:
	"""
    Interface đơn giản trả về chi phí tối thiểu
	"""
	cost, _ = hungarian_algorithm(cost_matrix)
	return cost


