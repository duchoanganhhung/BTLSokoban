from collections import deque
import numpy as np
from .utils import find_boxes_and_goals, is_deadlock

import random
import time  # dùng trong block __main__ (testing)

# ------------------------------
# ZOBRIST HASHING HELPERS
# ------------------------------
# Zobrist hashing: tạo bảng số ngẫu nhiên 64-bit cho mỗi ô và mỗi loại "vật thể"
#  - 1 bảng cho player (vị trí player)
#  - 1 bảng cho box (vị trí mỗi hộp)
# Khi xây hash: XOR tất cả giá trị tương ứng (player + tất cả các hộp).
# Lợi ích: cập nhật hash cực nhanh khi chỉ thay đổi một vài vị trí (XOR lại các entry cũ và mới).
# NOTE: seed được đặt cố định (12345) để reproducible giữa các lần chạy (tiện debug).

def init_zobrist(height, width):
    """
    Khởi tạo các bảng Zobrist.
    Trả về:
      zobrist_player: list length = height * width, mỗi phần tử là 64-bit random
      zobrist_box:    list length = height * width, mỗi phần tử là 64-bit random
    """
    zobrist_player = [random.getrandbits(64) for _ in range(height * width)]
    zobrist_box = [random.getrandbits(64) for _ in range(height * width)]
    return zobrist_player, zobrist_box

def zobrist_hash(player_pos, boxes, zobrist_player, zobrist_box, width):
    """
    Tạo zobrist hash cho trạng thái hiện tại.
    - player_pos: (x, y) tuple (vị trí player)
    - boxes: iterable các tuple (x, y) vị trí hộp (có thể là list/tuple/frozenset)
    - zobrist_player, zobrist_box: bảng đã khởi tạo
    - width: chiều rộng board (để chuyển (x,y) -> index 1D)
    Trả về 64-bit integer (Python int lưu vô hạn nhưng giá trị này là 64-bit).
    """
    px, py = player_pos
    # bắt đầu từ giá trị tương ứng với vị trí player
    h = zobrist_player[px * width + py]
    # XOR lần lượt vị trí các hộp
    for (x, y) in boxes:
        h ^= zobrist_box[x * width + y]
    return h

def zobrist_update(hash_val, old_player, new_player,
                   moved_box_from, moved_box_to,
                   zobrist_player, zobrist_box, width):
    """
    Cập nhật incremental cho zobrist hash khi player di chuyển (và có thể đẩy hộp).
    - hash_val: giá trị hash hiện tại (trước khi di chuyển)
    - old_player, new_player: (x,y) cũ và mới của player
    - moved_box_from, moved_box_to: (x,y) của box bị di chuyển (None nếu không có hộp bị đẩy)
    - các bảng zobrist và width như trên
    Kỹ thuật: XOR vị trí cũ và vị trí mới của player; nếu có box bị đẩy thì XOR vị trí cũ và mới của box.
    Vì XOR là phép nghịch đảo chính nó, việc XOR ra vào như vậy "loại" các contribution cũ và thêm contribution mới.
    """
    h = hash_val
    # loại contribution của vị trí player cũ và thêm contribution của vị trí player mới
    h ^= zobrist_player[old_player[0] * width + old_player[1]]
    h ^= zobrist_player[new_player[0] * width + new_player[1]]

    # nếu có đẩy hộp, loại contribution ở chỗ cũ rồi thêm ở chỗ mới
    if moved_box_from is not None:
        h ^= zobrist_box[moved_box_from[0] * width + moved_box_from[1]]
        h ^= zobrist_box[moved_box_to[0] * width + moved_box_to[1]]
    return h

# ------------------------------
# BFS with Zobrist
# ------------------------------
# Mục tiêu: BFS trên không gian trạng thái (player position + box configuration)
# - Sử dụng Zobrist hash để phát hiện trạng thái đã thăm (seen) một cách nhanh và tiết kiệm bộ nhớ
# - Mỗi state queue lưu: (player_pos, boxes_frozenset, zobrist_hash, path_string)
# - path_string: sequence của các move 'U', 'D', 'L', 'R' (mỗi ký tự tượng trưng 1 bước của player)
# Lưu ý: đây là BFS "step-based" (player move từng ô). Phiên bản push-based (chỉ lưu trạng thái sau mỗi đẩy)
# có thể hiệu quả hơn, nhưng ở đây ta giữ logic tương thích với code gốc.

def bfs(matrix, player_pos, widget=None, visualizer=False):
    """
    Thực thi BFS với Zobrist hashing.
    - matrix: numpy.ndarray kích thước (height, width) chứa ký tự biểu diễn map
              ký tự theo quy ước trong repo: '+', '-', '@', 'X', '$', '*', '%'...
    - player_pos: tuple (x, y) vị trí player khởi đầu
    - widget, visualizer: truyền qua nếu muốn hiển thị giao diện (giữ tham số tương thích)
    Trả về (path, depth) nếu tìm thấy; nếu không, (None, -1).
    """
    print('BFS with Zobrist Hashing')

    # lấy kích thước map
    shape = matrix.shape
    height, width = shape

    # tìm vị trí hộp và goal ban đầu từ ma trận (sử dụng helper trong utils)
    # find_boxes_and_goals nhận state ở dạng chuỗi phẳng -> ta cung cấp ''.join(matrix.flatten())
    boxes, goals, _ = find_boxes_and_goals(''.join(matrix.flatten()), shape)

    # khởi tạo bảng zobrist
    zobrist_player, zobrist_box = init_zobrist(height, width)

    # khởi tạo hash ban đầu (player_pos là (x,y), boxes là list of tuples)
    start_hash = zobrist_hash(player_pos, boxes, zobrist_player, zobrist_box, width)

    # start_state lưu như reference (không dùng trực tiếp trong seen, seen chỉ chứa hash)
    # tuy nhiên ta tạo start_state tuple để push vào queue (player_index, boxes_set)
    start_state = (player_pos[0] * width + player_pos[1], frozenset(boxes))

    # seen: set các zobrist hash đã thăm (lưu int 64-bit)
    # Lưu ý: Zobrist có khả năng va chạm (collision), nhưng xác suất cực nhỏ (2^-64).
    # Nếu muốn hoàn toàn an toàn, có thể lưu cả (hash, boxes_frozenset) tuple trong seen.
    seen = {start_hash}

    # queue BFS (deque) chứa các trạng thái cần duyệt:
    # mỗi phần tử: (player_pos (x,y), frozenset(box positions as tuples), zobrist_hash, path_string)
    q = deque([(player_pos, frozenset(boxes), start_hash, '')])

    # định nghĩa các move: (dx, dy)
    moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    # mapping move -> ký tự path
    direction = {(1, 0): 'D', (-1, 0): 'U', (0, -1): 'L', (0, 1): 'R'}

    # BFS loop
    while q:
        player, boxes, hash_val, path = q.popleft()
        # player là tuple (x, y)
        x, y = player

        # duyệt các hướng di chuyển
        for dx, dy in moves:
            nx, ny = x + dx, y + dy               # ô player sẽ di chuyển tới
            bx, by = nx + dx, ny + dy             # nếu player đẩy hộp thì ô đích của hộp

            # bảo vệ rìa: nếu truy cập ngoài bounds, matrix[nx, ny] sẽ raise IndexError
            # giả sử map có padding tường xung quanh, nhưng vẫn an toàn kiểm tra:
            if not (0 <= nx < height and 0 <= ny < width):
                # ngoài board -> skip
                continue

            # nếu ô target là tường thì player không thể di chuyển tới -> skip
            if matrix[nx, ny] == '+':
                continue

            # sao chép set hộp từ frozenset để dễ thay đổi (set mutable)
            new_boxes = set(boxes)
            new_player = (nx, ny)
            moved_box_from = moved_box_to = None

            # Nếu ô target chứa hộp (tức player đang cố đẩy)
            if (nx, ny) in boxes:
                # kiểm tra ô phía sau hộp phải hợp lệ
                # bounds check cho ô đẩy box
                if not (0 <= bx < height and 0 <= by < width):
                    continue
                # nếu ô đích của hộp là tường hoặc có hộp khác -> không thể đẩy
                if matrix[bx, by] == '+' or (bx, by) in boxes:
                    continue
                # cập nhật cấu hình hộp: di chuyển hộp từ (nx,ny) -> (bx,by)
                new_boxes.remove((nx, ny))
                new_boxes.add((bx, by))
                moved_box_from, moved_box_to = (nx, ny), (bx, by)

            # nếu không có hộp ở ô target, player chỉ di chuyển bình thường (không thay đổi hộp)

            # update zobrist hash một cách incremental: dùng zobrist_update để tránh phải rebuild hash toàn phần
            new_hash = zobrist_update(
                hash_val,                    # hash hiện tại
                player,                      # vị trí player cũ
                new_player,                  # vị trí player mới
                moved_box_from,              # nếu có box bị đẩy: vị trí cũ
                moved_box_to,                # nếu có box bị đẩy: vị trí mới
                zobrist_player,
                zobrist_box,
                width
            )

            # nếu hash đã thấy -> trạng thái đã thăm -> skip
            if new_hash in seen:
                continue
            # đánh dấu là đã thấy
            seen.add(new_hash)

            # thêm ký tự move vào path
            new_path = path + direction[(dx, dy)]

            # kiểm tra thắng: tất cả goal phải có box
            # `goals` là danh sách (x,y) ban đầu được trả từ find_boxes_and_goals
            # Chú ý: goals ở đây được lấy từ trạng thái ban đầu; không thay đổi theo thời gian
            if all(pos in new_boxes for pos in goals):
                # giải xong, in kết quả và trả về path và độ sâu (số bước)
                print(f'[BFS-ZOBRIST] Solved in {len(new_path)} moves: {new_path}')
                return new_path, len(new_path)

            # nếu chưa giải: đẩy trạng thái mới vào queue
            q.append((new_player, frozenset(new_boxes), new_hash, new_path))

    # nếu duyệt hết queue mà không tìm được giải
    print('[BFS-ZOBRIST] No solution found.')
    return None, -1


def solve_bfs(puzzle, widget=None, visualizer=False):
    """
    Wrapper để tương thích với API cũ:
    - puzzle: numpy matrix
    Trả về kết quả hàm bfs sau khi lấy vị trí player từ puzzle.
    """
    matrix = puzzle
    # tìm vị trí player ('*' hoặc '%' nếu player đứng trên goal)
    where = np.where((matrix == '*') | (matrix == '%'))
    player_pos = (where[0][0], where[1][0])
    return bfs(matrix, player_pos, widget, visualizer)


# block để test file trực tiếp (chạy như module)
if __name__ == '__main__':
    start = time.time()
    # load levels/lvl7.dat theo định dạng file (mỗi ô 1 ký tự)
    root = solve_bfs(np.loadtxt('levels/lvl7.dat', dtype='<U1'))
    print(f'Runtime: {time.time() - start} seconds')
