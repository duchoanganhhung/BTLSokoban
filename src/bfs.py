import time
from collections import deque

import numpy as np
import pygame

from .utils import can_move, get_state, is_deadlock, is_solved, print_state
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update  # <- mới


# def bfs(matrix, player_pos, widget=None, visualizer=False):
# 	print('Breadth-First Search')
# 	initial_state = get_state(matrix)
# 	shape = matrix.shape
# 	print_state(initial_state, shape)
# 	seen = {None}
# 	q = deque([(initial_state, player_pos, 0, '')])
# 	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
# 	curr_depth = 0
# 	direction = {
# 		(1, 0): 'D',
# 		(-1, 0): 'U', 
# 		(0, -1): 'L',
# 		(0, 1): 'R',
# 	}
# 	while q:
# 		if widget:
# 			pygame.event.pump()
# 		state, pos, depth, path = q.popleft()
# 		# if depth != curr_depth:
# 		# 	print(f'Depth: {depth}')
# 		# 	curr_depth = depth
# 		seen.add(state)
# 		for move in moves:
# 			new_state, _ = can_move(state, shape, pos, move)
# 			deadlock = is_deadlock(new_state, shape)
# 			if new_state in seen or deadlock:
# 				continue
# 			q.append((
# 				new_state, 
# 				(pos[0] + move[0], pos[1] + move[1]),
# 				depth + 1,
# 				path + direction[move],
# 			))
# 			if is_solved(new_state):
# 				print(f'[BFS] Solution found!\n\n{path + direction[move]}\nDepth {depth + 1}\n')
# 				if widget and visualizer:
# 					widget.solved = True
# 					widget.set_text(f'[BFS] Solution Found!\n{path + direction[move]}', 20)
# 					pygame.display.update()
# 				return (path + direction[move], depth + 1)
# 			if widget and visualizer:
# 				widget.set_text(f'[BFS] Solution Depth: {depth + 1}\n{path + direction[move]}', 20)
# 				pygame.display.update()
# 	print(f'[BFS] Solution not found!\n')
# 	if widget and visualizer:
# 		widget.set_text(f'[BFS] Solution Not Found!\nDepth {depth + 1}', 20)
# 		pygame.display.update()
# 	return (None, -1 if not q else depth + 1)


def solve_bfs(puzzle, widget=None, visualizer=False):
	matrix = puzzle
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]
	return bfs(matrix, player_pos, widget, visualizer)

	
if __name__ == '__main__':
	start = time.time()
	root = solve_bfs(np.loadtxt('levels/lvl7.dat', dtype='<U1'))
	print(f'Runtime: {time.time() - start} seconds')

from collections import deque
import numpy as np
import pygame

from .utils import get_state, is_deadlock, is_solved, print_state, can_move_delta  # <- thêm can_move_delta
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update  # <- mới

def bfs(matrix, player_pos, widget=None, visualizer=False):
	print('Breadth-First Search')
	initial_state = get_state(matrix)
	shape = matrix.shape
	print_state(initial_state, shape)

	# --- NEW: board & encode ---
	board = BoardIndex(matrix)
	boxes_mask, player_idx = encode_state_from_str(initial_state, board)

	# --- NEW: zobrist ---
	z = Zobrist(board.num)
	zkey = zobrist_initial(z, boxes_mask, player_idx)

	# --- NEW: seen là hash table các khóa nhỏ ---
	seen = {(zkey, boxes_mask, player_idx)}

	q = deque([(initial_state, player_pos, 0, '', zkey, boxes_mask, player_idx)])

	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

	while q:
		if widget:
			pygame.event.pump()
		state, pos, depth, path, zkey, boxes_mask, player_idx = q.popleft()

		for move in moves:
			res = can_move_delta(state, shape, pos, move, board)
			new_state, _mc, new_pos, old_pidx, new_pidx, box_delta = res

			if not new_state:
				continue

			# deadlock check vẫn dùng chuỗi như cũ (giữ nguyên logic)
			if is_deadlock(new_state, shape):
				continue

			# --- NEW: cập nhật khóa O(1) ---
			moved_from_idx, moved_to_idx = (None, None) if box_delta is None else box_delta

			new_boxes_mask = boxes_mask
			if moved_from_idx is not None:
				new_boxes_mask &= ~(1 << moved_from_idx)
			if moved_to_idx is not None:
				new_boxes_mask |= (1 << moved_to_idx)

			new_zkey = zobrist_update(
				z, zkey, old_pidx, new_pidx, moved_from_idx, moved_to_idx
			)

			key = (new_zkey, new_boxes_mask, new_pidx)
			if key in seen:
				continue
			seen.add(key)

			new_path = path + direction[move]
			q.append((new_state, new_pos, depth + 1, new_path, new_zkey, new_boxes_mask, new_pidx))

			if is_solved(new_state):
				print(f'[BFS] Solution found!\n\n{new_path}\nDepth {depth + 1}\n')
				if widget and visualizer:
					widget.solved = True
					widget.set_text(f'[BFS] Solution Found!\n{new_path}', 20)
					pygame.display.update()
				return (new_path, depth + 1)

			if widget and visualizer:
				widget.set_text(f'[BFS] Solution Depth: {depth + 1}\n{new_path}', 20)
				pygame.display.update()

	print(f'[BFS] Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'[BFS] Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not q else depth + 1)
