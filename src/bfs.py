import time
from collections import deque

import numpy as np
import pygame

# Import các hàm xử lý trạng thái và logic từ các file hỗ trợ
from .utils import can_move, get_state, is_deadlock, is_solved, print_state
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update  


def solve_bfs(puzzle, widget=None, visualizer=False):
    """
    Hàm tiện lợi để chạy BFS solver.
    - puzzle: ma trận numpy.ndarray của màn chơi Sokoban.
    - widget: giao diện pygame nếu có (dùng để hiển thị).
    - visualizer: bật/tắt chế độ hiển thị.
    """
    matrix = puzzle
    # Tìm vị trí người chơi (* hoặc %)
    where = np.where((matrix == '*') | (matrix == '%'))
    player_pos = where[0][0], where[1][0]
    return bfs(matrix, player_pos, widget, visualizer)


# Khi chạy độc lập file này
if __name__ == '__main__':
    start = time.time()
    root = solve_bfs(np.loadtxt('levels/lvl7.dat', dtype='<U1'))  # Load level 7 từ file
    print(f'Runtime: {time.time() - start} seconds')

# =================== BFS CORE ===================
from collections import deque
import numpy as np
import pygame

from .utils import get_state, is_deadlock, is_solved, print_state, can_move_delta  # dùng can_move_delta để sinh nước đi hợp lệ
from .state_codec import BoardIndex, encode_state_from_str, Zobrist, zobrist_initial, zobrist_update  # hash trạng thái


def bfs(matrix, player_pos, widget=None, visualizer=False):
    print('Breadth-First Search')
    initial_state = get_state(matrix)  # Chuỗi biểu diễn trạng thái ban đầu
    shape = matrix.shape
    print_state(initial_state, shape)

    # --- Tạo ánh xạ board: (r, c) <-> index hợp lệ (dùng cho bitmask)
    board = BoardIndex(matrix)

    # --- Mã hóa trạng thái: boxes_mask (bitmask các hộp), player_idx (vị trí người dưới dạng index 1D)
    boxes_mask, player_idx = encode_state_from_str(initial_state, board)

    # --- Khởi tạo Zobrist hashing để tạo khóa 64-bit nhanh
    z = Zobrist(board.num)
    zkey = zobrist_initial(z, boxes_mask, player_idx)

    # --- Bộ nhớ đã duyệt: set các khóa trạng thái đã thăm
    seen = {(zkey, boxes_mask, player_idx)}

    # --- Hàng đợi BFS: (state_str, pos, depth, path_str, zkey, boxes_mask, player_idx)
    q = deque([(initial_state, player_pos, 0, '', zkey, boxes_mask, player_idx)])

    # Các hướng di chuyển: D U L R
    moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

    while q:
        if widget:
            pygame.event.pump()  # Đảm bảo không treo giao diện

        state, pos, depth, path, zkey, boxes_mask, player_idx = q.popleft()

        for move in moves:
            # Kiểm tra nước đi theo hướng 'move' (hợp lệ, có đẩy hộp không)
            res = can_move_delta(state, shape, pos, move, board)
            new_state, _mc, new_pos, old_pidx, new_pidx, box_delta = res

            if not new_state:
                continue  # Không di chuyển được

            # Kiểm tra deadlock nếu trạng thái mới là bế tắc
            if is_deadlock(new_state, shape):
                continue

            # --- Cập nhật bitmask hộp (boxes_mask) ---
            moved_from_idx, moved_to_idx = (None, None) if box_delta is None else box_delta
            new_boxes_mask = int(boxes_mask)

            if moved_from_idx is not None:
                new_boxes_mask &= ~(1 << int(moved_from_idx))  # Bỏ hộp cũ

            if moved_to_idx is not None:
                new_boxes_mask |= (1 << int(moved_to_idx))    # Thêm hộp mới

            # Cập nhật khóa zobrist mới (O(1))
            new_zkey = zobrist_update(
                z, zkey, old_pidx, new_pidx, moved_from_idx, moved_to_idx
            )

            # Nếu trạng thái đã duyệt -> bỏ qua
            key = (new_zkey, new_boxes_mask, new_pidx)
            if key in seen:
                continue
            seen.add(key)

            new_path = path + direction[move]

            # Đưa vào hàng đợi BFS
            q.append((new_state, new_pos, depth + 1, new_path, new_zkey, new_boxes_mask, new_pidx))

            # Kiểm tra trạng thái thắng
            if is_solved(new_state):
                print(f'[BFS] Solution found!\n\n{new_path}\nDepth {depth + 1}\n')
                if widget and visualizer:
                    widget.solved = True
                    widget.set_text(f'[BFS] Solution Found!\n{new_path}', 20)
                    pygame.display.update()
                return (new_path, depth + 1)

            # Cập nhật giao diện nếu bật chế độ visualizer
            if widget and visualizer:
                widget.set_text(f'[BFS] Solution Depth: {depth + 1}\n{new_path}', 20)
                pygame.display.update()

    # Nếu BFS không tìm thấy lời giải
    print(f'[BFS] Solution not found!\n')
    if widget and visualizer:
        widget.set_text(f'[BFS] Solution Not Found!\nDepth {depth + 1}', 20)
        pygame.display.update()
    return (None, -1 if not q else depth + 1)
