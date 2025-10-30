import time
from .zobrist import ZobristHasher
from collections import defaultdict, deque
from heapq import heappop, heappush

import numpy as np
import pygame

from .utils import (can_move_delta, dijkstra_sum, get_state, is_deadlock, is_solved,
                    manhattan_sum, print_state, find_boxes_and_goals)
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update
from .deadlock_manager import DeadlockManager
from .deadlock_build import RetroBuilderSimple  # tên file bạn dùng
def solve_astar(puzzle, widget=None, visualizer=False, heuristic='manhattan', macro=False):
	"""Solve puzzle using A*.

	If macro=True use macro-move A* (pushes as operators). Otherwise use step-by-step A*.
	"""
	matrix = puzzle
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]
	if macro:
		return astar_macro(matrix, player_pos, widget, visualizer, heuristic)
	return astar(matrix, player_pos, widget, visualizer, heuristic)

	
if __name__ == '__main__':
	start = time.time()
	solve_astar(np.loadtxt('levels/lvl5.dat', dtype='<U1'), heuristic='dijkstra')
	print(f'Runtime: {time.time() - start} seconds')

def astar(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]'
	shape = matrix.shape
	initial_state = get_state(matrix)
	print_state(initial_state, shape)

	# board & encode
	board = BoardIndex(matrix)
	deadlock_mgr = DeadlockManager(board)
	#cells = [idx for idx, (r,c) in enumerate(board.from_idx) if matrix[r,c] not in ('X','$','%') ]
	builder = RetroBuilderSimple(board, deadlock_mgr,cells = None,matrix = matrix)
	builder.build(max_k=3,max_comb_k=2,Pmax=2)
	print("[DL] patterns", deadlock_mgr.count_patterns())
	boxes_mask, player_idx = encode_state_from_str(initial_state, board)
	z = Zobrist(board.num)
	zkey = zobrist_initial(z, boxes_mask, player_idx)

	# cost khởi tạo
	curr_depth = 0
	if heuristic == 'manhattan':
		curr_cost = manhattan_sum(initial_state, player_pos, shape)
	else:
		distances = defaultdict(lambda: [])
		curr_cost = dijkstra_sum(initial_state, player_pos, shape, distances)

	# seen bằng khóa nhỏ
	seen = {(zkey, boxes_mask, player_idx)}

	heap = []
	heappush(heap, (0, curr_cost, initial_state, player_pos, curr_depth, '', zkey, boxes_mask, player_idx))

	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	while heap:
		if widget:
			pygame.event.pump()
		_, curr_cost, state, pos, depth, path, zkey, boxes_mask, player_idx = heappop(heap)

		for move in moves:
			res = can_move_delta(state, shape, pos, move, board)
			new_state, move_cost, new_pos, old_pidx, new_pidx, box_delta = res
			if not new_state:
				continue
			# if is_deadlock(new_state, shape):
			# 	continue

			moved_from_idx, moved_to_idx = (None, None) if box_delta is None else box_delta
   
			new_boxes_mask = int(boxes_mask)
			if moved_from_idx is not None:
				new_boxes_mask &= ~(1 << int(moved_from_idx))
			if moved_to_idx is not None:
				new_boxes_mask |= (1 << int(moved_to_idx))
			if deadlock_mgr.match_boxes(new_boxes_mask):
				continue
			new_zkey = zobrist_update(z, zkey, old_pidx, new_pidx, moved_from_idx, moved_to_idx)

			key = (new_zkey, new_boxes_mask, new_pidx)
			if key in seen:
				continue
			seen.add(key)

			if heuristic == 'manhattan':
				new_h = manhattan_sum(new_state, new_pos, shape)
			else:
				new_h = dijkstra_sum(new_state, new_pos, shape, distances)
				if new_h == float('inf'):
					continue

			new_path = path + direction[move]
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

	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)


def astar_macro(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	"""Macro-move A*: expand only pushes but return full micro move string."""
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]'
	shape = matrix.shape
	initial_state = get_state(matrix)
	print_state(initial_state, shape)

	# Zobrist hasher for box-only hashing (used for closed set in macro search)
	hasher = ZobristHasher(shape)

	def player_dist_grid(state_str: str, start: tuple[int, int]):
		"""BFS distances for the player; treats walls and boxes as obstacles."""
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
				if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
					continue
				ch = state_str[nr * w + nc]
				if ch in obstacles:
					continue
				nd = dist[r, c] + 1
				if nd < dist[nr, nc]:
					dist[nr, nc] = nd
					q.append((nr, nc))
		return dist

	def reconstruct_walk(dist, start, target):
		if dist[target] == float('inf'):
			return []
		sr, sc = start
		tr, tc = target
		steps = []
		r, c = tr, tc
		neighs = [(1, 0, 'D'), (-1, 0, 'U'), (0, -1, 'L'), (0, 1, 'R')]
		while (r, c) != (sr, sc):
			found = False
			for dr, dc, sym in neighs:
				pr, pc = r - dr, c - dc
				if 0 <= pr < shape[0] and 0 <= pc < shape[1] and dist[pr, pc] == dist[r, c] - 1:
					steps.append(sym)
					r, c = pr, pc
					found = True
					break
			if not found:
				return []
		steps.reverse()
		return steps

	# Initial heuristic
	if heuristic == 'manhattan':
		curr_h = manhattan_sum(initial_state, player_pos, shape)
	else:
		distances = defaultdict(lambda: [])
		curr_h = dijkstra_sum(initial_state, player_pos, shape, distances)

	# Closed set over boxes configuration (order-independent) using box-only Zobrist
	seen_boxes = {hasher.hash_boxes(initial_state)}

	heap = []
	# (f, h, state, pos, depth, path, g)
	heappush(heap, (curr_h, curr_h, initial_state, player_pos, 0, '', 0))

	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	while heap:
		if widget:
			pygame.event.pump()
		_, curr_h, state, pos, depth, path, curr_g = heappop(heap)

		# Compute reachability for current player position
		dist = player_dist_grid(state, pos)
		hgt, wid = shape

		# Enumerate all boxes
		boxes, _, boxes_on_goal = find_boxes_and_goals(state, shape)
		all_boxes = boxes + boxes_on_goal

		for (r, c) in all_boxes:
			for dx, dy in moves:
				br, bc = r - dx, c - dy  # behind (player stands here)
				ar, ac = r + dx, c + dy  # ahead (box moves here)

				# bounds
				if not (1 <= br < hgt - 1 and 1 <= bc < wid - 1):
					continue
				if not (1 <= ar < hgt - 1 and 1 <= ac < wid - 1):
					continue

				behind1d = br * wid + bc
				box1d = r * wid + c
				ahead1d = ar * wid + ac

				# player must reach behind
				if dist[br, bc] == float('inf'):
					continue
				# ahead must be free or goal
				if state[ahead1d] not in '-X':
					continue

				# Build new state after push
				new_state = list(state)

				# Clear old player tile
				cur1d = pos[0] * wid + pos[1]
				if new_state[cur1d] == '*':
					new_state[cur1d] = '-'
				elif new_state[cur1d] == '%':
					new_state[cur1d] = 'X'

				# Move box to ahead
				old_ahead = new_state[ahead1d]
				new_state[ahead1d] = '@' if old_ahead == '-' else '$'

				# Player ends up where the box was
				box_ch = new_state[box1d]
				new_state[box1d] = '*' if box_ch == '@' else '%'

				new_state = ''.join(new_state)
				new_pos = (r, c)

				# Deadlock pruning
				if is_deadlock(new_state, shape):
					continue

				# Use box-only Zobrist hash to detect equivalent box configurations
				new_boxes_hash = hasher.hash_boxes(new_state)
				if new_boxes_hash in seen_boxes:
					continue
				seen_boxes.add(new_boxes_hash)

				# Heuristic
				if heuristic == 'manhattan':
					new_h = manhattan_sum(new_state, new_pos, shape)
				else:
					new_h = dijkstra_sum(new_state, new_pos, shape, distances)
					if new_h == float('inf'):
						continue

				# Reconstruct micro walk path
				micro_steps = reconstruct_walk(dist, pos, (br, bc))
				walk_steps = len(micro_steps)
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

	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)