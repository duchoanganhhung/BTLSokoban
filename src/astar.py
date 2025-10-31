import time
from collections import defaultdict, deque
from heapq import heappop, heappush

import numpy as np
import pygame

from .utils import (can_move_delta, dijkstra_sum, get_state, is_deadlock, is_solved,
					manhattan_sum, minimum_matching_sum, print_state, find_boxes_and_goals)
from .state_codec import BoardIndex, encode_state_from_str, ZobristHash
from .deadlock_manager import DeadlockManager
from .deadlock_build import RetroBuilderSimple


# Module-level:
# Đây là triển khai A* (cùng với phiên bản macro-move) cho Sokoban.
# Bình luận được viết bằng tiếng Việt để giải thích từng phần:
# - solve_astar: entry point chọn giữa step-by-step A* và macro A* (push-based)
# - astar: A* tiêu chuẩn với state bao gồm vị trí người chơi và các hộp
# - astar_macro: A* chỉ mở rộng các thao tác push (mỗi node = một push),
#               nhưng kết quả trả về vẫn là chuỗi micro-move đầy đủ.


def solve_astar(puzzle, widget=None, visualizer=False, heuristic='manhattan', macro=False):
	"""
	Hàm khởi động giải bài toán Sokoban bằng A*.

	Tham số:
	- puzzle: ma trận numpy đại diện cho board (ký tự theo định dạng dự án)
	- widget: (tuỳ chọn) widget hiển thị để cập nhật tiến trình
	- visualizer: bật hiển thị động khi True
	- heuristic: tên heuristic ('manhattan', 'dijkstra', 'hungarian')
	- macro: nếu True dùng astar_macro (mỗi bước là một push)

	Trả về: theo định nghĩa của `astar` hoặc `astar_macro`:
	- (path, depth) nếu tìm được lời giải
	- (None, -1) nếu không tìm được
	"""
	matrix = puzzle
	# tìm vị trí người chơi trong ma trận nhập (ký hiệu '*' hoặc '%')
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]
	if macro:
		return astar_macro(matrix, player_pos, widget, visualizer, heuristic)
	return astar(matrix, player_pos, widget, visualizer, heuristic)


if __name__ == '__main__':
	# Cho phép chạy file độc lập để debug nhanh
	start = time.time()
	solve_astar(np.loadtxt('levels/lvl5.dat', dtype='<U1'), heuristic='dijkstra')
	print(f'Runtime: {time.time() - start} seconds')


def astar(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	"""
	A* triển khai theo từng bước (micro-moves):

	- Mỗi node của A* lưu trạng thái đầy đủ (chuỗi trạng thái), vị trí người chơi,
	  và một bitmask (boxes_mask) để mô tả vị trí các hộp (dùng cho cập nhật nhanh và
	  cho tính toán hash nhỏ gọn).
	- Dùng Zobrist để cập nhật hash nhanh khi người chơi hoặc hộp di chuyển.
	- DeadlockManager + RetroBuilderSimple được dùng để sinh/so sánh pattern deadlock;
	  nếu cấu hình hộp khớp pattern deadlock thì ta prune (loại nhánh) ngay.

	Lưu ý về heuristic:
	- 'manhattan': tổng khoảng cách Manhattan từ từng hộp tới mục tiêu tương ứng
	- 'dijkstra': sử dụng dijkstra_sum (tính tỉ mỉ hơn, có thể chậm hơn nhưng tốt hơn cho map có tường)
	- 'hungarian': giải bài toán gán tối ưu giữa boxes và goals (có cache để tránh tính lại)

	Trả về: (path, depth) như mô tả ở trên.
	"""
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]' if heuristic == 'dijkstra' else '[Hungarian]'
	shape = matrix.shape
	initial_state = get_state(matrix)
	print_state(initial_state, shape)

	# Chuẩn bị board và deadlock manager
	board = BoardIndex(matrix)
	deadlock_mgr = DeadlockManager(board)
	# RetroBuilderSimple: xây dựng các pattern deadlock (tham số có thể được tinh chỉnh)
	builder = RetroBuilderSimple(board, deadlock_mgr, cells=None, matrix=matrix)
	builder.build(max_k=3, max_comb_k=2, Pmax=2)
	print("[DL] patterns", deadlock_mgr.count_patterns())

	# Mã hóa trạng thái khởi đầu: boxes_mask là bitmask của hộp, player_idx vị trí index 1D
	boxes_mask, player_idx = encode_state_from_str(initial_state, board)
	# ZobristHash unified object for bitboard (delta) and grid hashing
	z = ZobristHash(num_cells=board.num, shape=shape)
	zkey = z.initial_key(boxes_mask, player_idx)

	# Giá trị f/g/h ban đầu
	curr_depth = 0
	if heuristic == 'manhattan':
		curr_cost = manhattan_sum(initial_state, player_pos, shape)
	elif heuristic == 'dijkstra':
		# distances: cache cho dijkstra để tránh tính lại nhiều lần
		distances = defaultdict(lambda: [])
		curr_cost = dijkstra_sum(initial_state, player_pos, shape, distances)
	elif heuristic == 'hungarian':
		# hung_cache được giữ suốt vòng lặp để tái sử dụng kết quả matching
		hung_cache = {}
		curr_cost = minimum_matching_sum(initial_state, player_pos, shape, hung_cache)
	else:
		curr_cost = manhattan_sum(initial_state, player_pos, shape)  # fallback

	# closed set (seen) dùng tuple (zkey, boxes_mask, player_idx) để đảm bảo thứ tự không quan trọng
	seen = {(zkey, boxes_mask, player_idx)}

	# heap lưu các node theo thứ tự ưu tiên; tuple lưu (g, h, state, pos, depth, path, zkey, boxes_mask, player_idx)
	heap = []
	heappush(heap, (0, curr_cost, initial_state, player_pos, curr_depth, '', zkey, boxes_mask, player_idx))

	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	while heap:
		if widget:
			# giữ cho event queue của pygame không bị treo khi đang hiển thị
			pygame.event.pump()
		_, curr_cost, state, pos, depth, path, zkey, boxes_mask, player_idx = heappop(heap)

		# Duyệt 4 hướng di chuyển
		for move in moves:
			# can_move_delta trả về nhiều thông tin: new_state, chi phí move, vị trí mới, chỉ số player cũ/mới, thay đổi hộp
			res = can_move_delta(state, shape, pos, move, board)
			new_state, move_cost, new_pos, old_pidx, new_pidx, box_delta = res
			if not new_state:
				continue

			# box_delta = (from_idx, to_idx) nếu có push, ngược lại None
			moved_from_idx, moved_to_idx = (None, None) if box_delta is None else box_delta

			# cập nhật bitmask hộp nhanh bằng bit operations
			new_boxes_mask = int(boxes_mask)
			if moved_from_idx is not None:
				new_boxes_mask &= ~(1 << int(moved_from_idx))
			if moved_to_idx is not None:
				new_boxes_mask |= (1 << int(moved_to_idx))

			# Deadlock pruning: nếu cấu hình hộp khớp pattern deadlock, bỏ qua
			if deadlock_mgr.match_boxes(new_boxes_mask):
				continue

			# cập nhật zobrist key cho node mới (rất nhanh, tránh hash toàn bộ state)
			new_zkey = z.update_key(zkey, old_pidx, new_pidx, moved_from_idx, moved_to_idx)

			key = (new_zkey, new_boxes_mask, new_pidx)
			if key in seen:
				continue
			seen.add(key)

			# tính heuristic mới
			if heuristic == 'manhattan':
				new_h = manhattan_sum(new_state, new_pos, shape)
			elif heuristic == 'dijkstra':
				new_h = dijkstra_sum(new_state, new_pos, shape, distances)
				if new_h == float('inf'):
					# unreachable configuration
					continue
			elif heuristic == 'hungarian':
				new_h = minimum_matching_sum(new_state, new_pos, shape, hung_cache)
				if new_h == float('inf'):
					continue
			else:
				new_h = manhattan_sum(new_state, new_pos, shape)

			new_path = path + direction[move]
			# lưu node mới vào heap (g + previous g, h, ...)
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

			# Kiểm tra giải
			if is_solved(new_state):
				print(f'{heur} Solution found!\n\n{new_path}\nDepth {depth + 1}\n')
				if widget and visualizer:
					widget.solved = True
					widget.set_text(f'{heur} Solution Found!\n{new_path}', 20)
					pygame.display.update()
				return (new_path, depth + 1)

			# cập nhật widget nếu cần
			if widget and visualizer:
				widget.set_text(f'{heur} Solution Depth: {depth + 1}\n{new_path}', 20)
				pygame.display.update()

	# Không tìm được lời giải
	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)


def astar_macro(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	"""
	A* ở mức macro (mỗi action là một push).

	Ý tưởng:
	- Thay vì mở rộng mọi micro-move (player từng bước), ta chỉ mở rộng những hành động push hợp lệ.
	- Mỗi node biểu diễn một cấu hình state sau một push; closed-set chỉ cần lưu cấu hình box
	  (vì thứ tự player giữa các push không quan trọng cho box-only equivalence).
	- Để xây dựng đường đi micro đầy đủ, ta giữ lại BFS từ vị trí player tới vị trí đứng sau hộp
	  (player_dist_grid) và reconstruct_walk để tái tạo chuỗi micro-steps.

	Input/Output giống `astar`.
	"""
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]' if heuristic == 'dijkstra' else '[Hungarian]'
	shape = matrix.shape
	initial_state = get_state(matrix)
	print_state(initial_state, shape)

	# ZobristHash chỉ cho các box (bỏ qua vị trí player) -> dùng cho closed set nhanh
	hasher = ZobristHash(shape=shape)

	def player_dist_grid(state_str: str, start: tuple[int, int]):
		"""
		BFS đơn giản trả về ma trận khoảng cách từ start tới mọi ô mà player có thể tới.

		Xử lý:
		- Tường/ngoại biên và ô có hộp ('@' hoặc '$') được coi là chướng ngại.
		- Trả về mảng 2D với khoảng cách (float('inf') cho ô không tới được).
		"""
		h, w = shape
		inf = float('inf')
		dist = np.full((h, w), inf)
		dist[start] = 0
		q = [(start[0], start[1])]
		head = 0
		obstacles = set(['+', '@', '$'])
		while head < len(q):
			r, c = q[head]
			head += 1
			for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
				nr, nc = r + dr, c + dc
				# chỉ xét trong bounds hợp lệ (loại bỏ ngoại biên tường)
				if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
					continue
				ch = state_str[nr * w + nc]
				if ch in obstacles:
					continue
				nd = dist[r, c] + 1
				# Nếu khoảng cách tới ô mới nhỏ hơn khoảng cách hiện tại
				if nd < dist[nr, nc]:
					# Cập nhật khoảng cách tới ô mới
					dist[nr, nc] = nd
					# Thêm ô mới vào hàng đợi BFS
					q.append((nr, nc))
		# Trả về ma trận khoảng cách từ vị trí xuất phát
		return dist

	def reconstruct_walk(dist, start, target):
		"""
		Dựa vào ma trận khoảng cách `dist` tái tạo đường đi ngắn nhất (theo BFS) từ start tới target.
		Trả về danh sách ký tự 'UDLR'. Nếu target unreachable trả về list rỗng.
		"""
		# Nếu target không thể tới được (khoảng cách vô cùng)
		if dist[target] == float('inf'):
			# Trả về list rỗng
			return []
		# Giải nén tọa độ xuất phát
		sr, sc = start
		# Giải nén tọa độ đích
		tr, tc = target
		# Danh sách lưu các bước di chuyển (U/D/L/R)
		steps = []
		# Vị trí hiện tại bắt đầu từ target
		r, c = tr, tc
		# Danh sách 4 hướng lân cận: (delta_row, delta_col, ký_tự_hướng)
		neighs = [(1, 0, 'D'), (-1, 0, 'U'), (0, -1, 'L'), (0, 1, 'R')]
		# Lặp cho tới khi quay lại vị trí xuất phát
		while (r, c) != (sr, sc):
			# Cờ đánh dấu tìm thấy hướng tiếp theo
			found = False
			# Duyệt 4 hướng lân cận
			for dr, dc, sym in neighs:
				# Tính tọa độ ô trước đó (đi ngược lại)
				pr, pc = r - dr, c - dc
				# Kiểm tra ô trước đó có hợp lệ và khoảng cách giảm 1 (là bước tiếp theo về phía start)
				if 0 <= pr < shape[0] and 0 <= pc < shape[1] and dist[pr, pc] == dist[r, c] - 1:
					# Thêm ký tự hướng vào danh sách bước
					steps.append(sym)
					# Di chuyển vị trí hiện tại tới ô trước đó
					r, c = pr, pc
					found = True
					break
			if not found:
				# Không thể reconstruct (nếu dữ liệu dist bị bất thường)
				return []
		steps.reverse()
		return steps

	# Heuristic khởi tạo + caches cần thiết
	distances = defaultdict(lambda: [])
	if heuristic == 'manhattan':
		curr_h = manhattan_sum(initial_state, player_pos, shape)
		hung_cache = None
	elif heuristic == 'dijkstra':
		curr_h = dijkstra_sum(initial_state, player_pos, shape, distances)
		hung_cache = None
	elif heuristic == 'hungarian':
		hung_cache = {}
		curr_h = minimum_matching_sum(initial_state, player_pos, shape, hung_cache)
		# Nếu minimum_matching_sum trả về float('inf') nghĩa là không thể tìm được
		# một ghép hợp lệ giữa các hộp và mục tiêu (ví dụ do một ô không tới được),
		# ta không dùng giá trị vô hạn cho heuristic vì sẽ phá vỡ so sánh trong A*.
		# Do đó fallback về heuristic đơn giản (Manhattan) để tiếp tục tìm kiếm.
		if curr_h == float('inf'):
			curr_h = manhattan_sum(initial_state, player_pos, shape)
	else:
		curr_h = manhattan_sum(initial_state, player_pos, shape)
		hung_cache = None

	# closed set dựa trên cấu hình box (bỏ qua vị trí player)
	seen_boxes = {hasher.hash_boxes(initial_state)}

	heap = []
	# heap item: (f, h, state, pos, depth, path, g)
	heappush(heap, (curr_h, curr_h, initial_state, player_pos, 0, '', 0))

	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	while heap:
		if widget:
			pygame.event.pump()
		_, curr_h, state, pos, depth, path, curr_g = heappop(heap)

		# Tính reachability cho player từ vị trí hiện tại
		dist = player_dist_grid(state, pos)
		hgt, wid = shape

		# Lấy danh sách tất cả các hộp (bao gồm hộp trên goal)
		boxes, _, boxes_on_goal = find_boxes_and_goals(state, shape)
		all_boxes = boxes + boxes_on_goal

		# Với mỗi hộp, xét 4 hướng push
		for (r, c) in all_boxes:
			for dx, dy in moves:
				br, bc = r - dx, c - dy  # ô đứng phía sau hộp (player phải tới đây để push)
				ar, ac = r + dx, c + dy  # ô phía trước hộp (hộp sẽ di chuyển tới đây)

				# Kiểm tra bounds (loại bỏ rìa tường)
				if not (1 <= br < hgt - 1 and 1 <= bc < wid - 1):
					continue
				if not (1 <= ar < hgt - 1 and 1 <= ac < wid - 1):
					continue

				# Chuyển đổi tọa độ 2D sang chỉ số 1D trong chuỗi state
				# behind1d: vị trí của ô mà player cần đứng để push hộp (phía sau hộp)
				behind1d = br * wid + bc
				# box1d: vị trí hiện tại của hộp trong chuỗi state
				box1d = r * wid + c
				# ahead1d: vị trí mà hộp sẽ di chuyển tới sau khi được push (phía trước hộp)
				ahead1d = ar * wid + ac

				# Player phải tới được ô phía sau
				if dist[br, bc] == float('inf'):
					continue
				# ô phía trước phải là free ('-' ) hoặc goal ('X') để push vào
				if state[ahead1d] not in '-X':
					continue

				# Xây dựng trạng thái mới sau push (không cần mô phỏng micro walk ở đây)
				new_state = list(state)

				# Xóa kí hiệu player ở vị trí cũ (biến thành floor hoặc goal tuỳ trước đó)
				cur1d = pos[0] * wid + pos[1]
				if new_state[cur1d] == '*':
					new_state[cur1d] = '-'
				elif new_state[cur1d] == '%':
					new_state[cur1d] = 'X'

				# move box tới ahead1d
				old_ahead = new_state[ahead1d]
				new_state[ahead1d] = '@' if old_ahead == '-' else '$'

				# player kết thúc ở vị trí box trước push
				box_ch = new_state[box1d]
				new_state[box1d] = '*' if box_ch == '@' else '%'

				new_state = ''.join(new_state)
				new_pos = (r, c)

				# Deadlock pruning nhanh
				if is_deadlock(new_state, shape):
					continue

				# Hash box-only để tránh revisit các cấu hình tương đương
				new_boxes_hash = hasher.hash_boxes(new_state)
				if new_boxes_hash in seen_boxes:
					continue
				seen_boxes.add(new_boxes_hash)

				# Heuristic cho node mới
				if heuristic == 'manhattan':
					new_h = manhattan_sum(new_state, new_pos, shape)
				elif heuristic == 'dijkstra':
					new_h = dijkstra_sum(new_state, new_pos, shape, distances)
					if new_h == float('inf'):
						continue
				elif heuristic == 'hungarian':
					new_h = minimum_matching_sum(new_state, new_pos, shape, hung_cache)
					if new_h == float('inf'):
						continue

				# Tái tạo chuỗi micro-steps cho player đi tới ô phía sau (br,bc)
				micro_steps = reconstruct_walk(dist, pos, (br, bc))
				walk_steps = len(micro_steps)
				# Chi phí được tính tuỳ theo design: 3 * bước đi + 2 cho push (hoặc 0 nếu push vào goal)
				walk_cost = 3 * walk_steps
				push_cost = 0 if old_ahead == 'X' else 2
				move_cost = walk_cost + push_cost

				new_path = path + ''.join(micro_steps) + direction[(dx, dy)]

				new_g = curr_g + move_cost
				new_f = new_g + new_h

				heappush(heap, (
					new_f,
					new_h,
					new_state,
					new_pos,
					depth + 1,
					new_path,
					new_g,
				))

				if is_solved(new_state):
					print(f'{heur} Solution found!\n\n{new_path}\nDepth {depth + 1}\n')
					if widget and visualizer:
						widget.solved = True
						widget.set_text(f'{heur} Solution Found!\n{new_path}', 20)
						pygame.display.update()
					return (new_path, depth + 1)

				if widget and visualizer:
					widget.set_text(f'{heur} Solution Depth: {depth + 1}\n{new_path}', 20)
					pygame.display.update()

	# Không tìm thấy solution
	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)