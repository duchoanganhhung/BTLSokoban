import time
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
import pygame

# Import các hàm tiện ích từ các module khác trong project
from .utils import (can_move_delta, dijkstra_sum, get_state, is_deadlock, is_solved,
                    manhattan_sum, print_state)
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update
from .deadlock_manager import DeadlockManager
from .deadlock_build import RetroBuilderSimple  # bộ sinh deadlock tự động

# ================== HÀM GIẢI QUYẾT CHÍNH ==================
def solve_astar(puzzle, widget=None, visualizer=False, heuristic='manhattan'):
	"""
	Hàm tiện ích gọi thuật toán A*.
	- puzzle: ma trận Sokoban (numpy.ndarray)
	- heuristic: 'manhattan' hoặc 'dijkstra'
	"""
	matrix = puzzle
	# tìm vị trí người chơi (* hoặc %)
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]
	# gọi thuật toán A* chính
	return astar(matrix, player_pos, widget, visualizer, heuristic)

	
if __name__ == '__main__':
	# Khi chạy trực tiếp file này
	start = time.time()
	# Load map từ file lvl5.dat và chạy A* (Dijkstra heuristic)
	solve_astar(np.loadtxt('levels/lvl5.dat', dtype='<U1'), heuristic='dijkstra')
	print(f'Runtime: {time.time() - start} seconds')


# ================== THUẬT TOÁN A* ==================
def astar(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]'
	shape = matrix.shape

	# Lấy trạng thái ban đầu của bàn chơi (chuỗi phẳng H*W)
	initial_state = get_state(matrix)
	print_state(initial_state, shape)

	# --- Khởi tạo board và bộ quản lý deadlock ---
	board = BoardIndex(matrix)  # ánh xạ (r,c) <-> index hợp lệ (bitmask)
	deadlock_mgr = DeadlockManager(board)

	# Sinh tự động các pattern deadlock từ ma trận hiện tại
	builder = RetroBuilderSimple(board, deadlock_mgr, cells=None, matrix=matrix)
	builder.build(max_k=3, max_comb_k=2, Pmax=2)
	print("[DL] patterns", deadlock_mgr.count_patterns())

	# Mã hóa trạng thái ban đầu thành (mask hộp, vị trí người)
	boxes_mask, player_idx = encode_state_from_str(initial_state, board)

	# Tạo bảng Zobrist và khóa hash ban đầu
	z = Zobrist(board.num)
	zkey = zobrist_initial(z, boxes_mask, player_idx)

	# --- Khởi tạo chi phí ---
	curr_depth = 0
	if heuristic == 'manhattan':
		curr_cost = manhattan_sum(initial_state, player_pos, shape)
	else:
		distances = defaultdict(lambda: [])
		curr_cost = dijkstra_sum(initial_state, player_pos, shape, distances)

	# --- Bảng trạng thái đã thăm (hash nhỏ) ---
	# Lưu bằng tuple (zkey, boxes_mask, player_idx)
	seen = {(zkey, boxes_mask, player_idx)}

	# --- Khởi tạo hàng đợi ưu tiên (min-heap) cho A* ---
	# Mỗi phần tử: (tổng_cost, heuristic, state, pos, depth, path, zkey, boxes_mask, player_idx)
	heap = []
	heappush(heap, (0, curr_cost, initial_state, player_pos, curr_depth, '', zkey, boxes_mask, player_idx))

	# Các hướng di chuyển
	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	# --- Bắt đầu vòng lặp A* ---
	while heap:
		if widget:
			pygame.event.pump()  # cập nhật giao diện nếu có
		# Lấy trạng thái có chi phí nhỏ nhất ra khỏi heap
		_, curr_cost, state, pos, depth, path, zkey, boxes_mask, player_idx = heappop(heap)

		# Duyệt từng hướng di chuyển
		for move in moves:
			# Hàm kiểm tra nước đi (delta): cho biết có đẩy được hộp không, đẩy tới đâu
			res = can_move_delta(state, shape, pos, move, board)
			new_state, move_cost, new_pos, old_pidx, new_pidx, box_delta = res
			if not new_state:
				continue

			# Xử lý cập nhật bitmask của hộp (nếu có di chuyển hộp)
			moved_from_idx, moved_to_idx = (None, None) if box_delta is None else box_delta

			new_boxes_mask = int(boxes_mask)
			if moved_from_idx is not None:
				new_boxes_mask &= ~(1 << int(moved_from_idx))  # xóa hộp cũ
			if moved_to_idx is not None:
				new_boxes_mask |= (1 << int(moved_to_idx))    # thêm hộp mới

			# Kiểm tra xem vị trí mới có rơi vào deadlock không
			if deadlock_mgr.match_boxes(new_boxes_mask):
				continue

			# Cập nhật khóa zobrist mới (O(1))
			new_zkey = zobrist_update(z, zkey, old_pidx, new_pidx, moved_from_idx, moved_to_idx)

			# Nếu trạng thái đã thăm thì bỏ qua
			key = (new_zkey, new_boxes_mask, new_pidx)
			if key in seen:
				continue
			seen.add(key)

			# Tính heuristic mới (Manhattan hoặc Dijkstra)
			if heuristic == 'manhattan':
				new_h = manhattan_sum(new_state, new_pos, shape)
			else:
				new_h = dijkstra_sum(new_state, new_pos, shape, distances)
				if new_h == float('inf'):
					continue  # trạng thái không tới được

			# Cộng chuỗi bước di chuyển
			new_path = path + direction[move]

			# Thêm trạng thái mới vào heap
			heappush(heap, (
				move_cost + curr_cost,
				new_h,
				new_state,
				new_pos,
				depth + 1,
				new_path,
				new_zkey,
				new_boxes_mask,
				new_pidx,
			))

			# Nếu đã giải xong thì thông báo và thoát
			if is_solved(new_state):
				print(f'{heur} Solution found!\n\n{new_path}\nDepth {depth + 1}\n')
				if widget and visualizer:
					widget.solved = True
					widget.set_text(f'{heur} Solution Found!\n{new_path}', 20)
					pygame.display.update()
				return (new_path, depth + 1)

			# Cập nhật giao diện nếu có chế độ visualizer
			if widget and visualizer:
				widget.set_text(f'{heur} Solution Depth: {depth + 1}\n{new_path}', 20)
				pygame.display.update()

	# Nếu duyệt hết mà không tìm thấy nghiệm
	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)
