# src/state_codec.py
import numpy as np
import random

WALL = '+'
BOXES = {'@', '$'}
PLAYER = {'*', '%'}

#Biến ma trận 2D thành dạng số hóa 1D bitmask 
class BoardIndex:
    """
    Map (r,c) -> idx chỉ số ô hợp lệ (không phải tường) để làm bitboard.
    """
    __slots__ = ('H','W','to_idx','from_idx','num')
    # Khỏi dùng __dict__, cấp sẵn đúng  5 biến
    def __init__(self, matrix: np.ndarray):
        H, W = matrix.shape
        self.H, self.W = H, W
        self.to_idx = -np.ones((H, W), dtype=int)
        #Ánh xạ tọa độ 2D sang chỉ số 1D bỏ tường, các chỉ số từ 0 đến n lá để chỉ các ô đi được
        idx = 0
        for r in range(H):
            for c in range(W):
                if matrix[r, c] != WALL:
                    self.to_idx[r, c] = idx
                    idx += 1
        self.num = idx
        self.from_idx = np.full((idx, 2), -1, dtype=int) # Tạo 1 mảng có kích thước(idx,2)
        it = np.where(self.to_idx != -1) #Trả về 2 mảng hàng cột ghép lại có giá trị khác -1
        for r, c in zip(it[0], it[1]):
            self.from_idx[self.to_idx[r, c]] = (r, c)

def encode_state_from_str(state: str, board: BoardIndex) -> tuple[int, int]:
    """
    Từ chuỗi state H*W → (boxes_mask:int, player_idx:int)
    """
    H, W = board.H, board.W
    boxes_mask = 0
    player_idx = -1
    for i, ch in enumerate(state):
        r, c = divmod(i, W)
        if ch == WALL:
            continue
        idx = board.to_idx[r, c]
        if ch in BOXES:
            boxes_mask = int(boxes_mask | (1 << int(idx)))
        if ch in PLAYER:
            player_idx = idx
    return boxes_mask, player_idx

# -------- ZOBRIST --------
#class Zobrist là “bộ máy băm siêu nhanh” giúp bạn mã hóa trạng thái Sokoban thành một mã 64-bit duy nhất, để kiểm tra trùng lặp (duplicate state) mà không cần so sánh cả ma trận.
class Zobrist:
    """
    Zobrist keys cho (boxes, player). Cập nhật O(1) theo delta.
    """
    __slots__ = ('box_rand','player_rand')
    def __init__(self, num_cells: int, seed: int = 1337):
        rnd = random.Random(seed)
        # 64-bit số ngẫu nhiên
        self.box_rand = [rnd.getrandbits(64) for _ in range(num_cells)]
        self.player_rand = [rnd.getrandbits(64) for _ in range(num_cells)]

def zobrist_initial(z: Zobrist, boxes_mask: int, player_idx: int) -> int:
    key = 0
    bm = boxes_mask
    idx = 0
    while bm:
        if bm & 1:
            key ^= z.box_rand[idx]
        bm >>= 1
        idx += 1
    if player_idx >= 0:
        key ^= z.player_rand[player_idx]
    return key
# def zobrist_initial(z: Zobrist, boxes_mask: int, player_idx: int) -> int:
#     key = 0
#     bm = boxes_mask
#     num_boxes = (int(boxes_mask) & 0xFFFFFFFFFFFFFFFF).bit_count()
#     print("Số hộp đang bật =", num_boxes)
#     idx = highest_bit_index(boxes_mask)
#     print(f"Bit cao nhất được bật là: {idx}")
#     idx = 0
#     while bm:
#         if bm & 1:
#             try:
#                 key ^= z.box_rand[idx]
#             except IndexError:
#                 print(f"[ERROR] IndexError at idx={idx}")
#                 print(f" - boxes_mask = {boxes_mask} (bin={bin(boxes_mask)})")
#                 print(f" - len(z.box_rand) = {len(z.box_rand)}")
#                 raise  # giữ nguyên lỗi để traceback hiện ra
#         bm >>= 1
#         idx += 1

#     if player_idx >= 0:
#         try:
#             key ^= z.player_rand[player_idx]
#         except IndexError:
#             print(f"[ERROR] player_idx={player_idx} out of range!")
#             print(f" - len(z.player_rand) = {len(z.player_rand)}")
#             raise
#     return key

def zobrist_update(
    z: Zobrist,
    key: int,
    old_player_idx: int,
    new_player_idx: int,
    moved_box_from_idx: int | None,
    moved_box_to_idx: int | None,
) -> int:
    # player XOR out/in
    if old_player_idx >= 0:
        key ^= z.player_rand[old_player_idx]
    if new_player_idx >= 0:
        key ^= z.player_rand[new_player_idx]
    # box XOR out/in nếu có đẩy hộp
    if moved_box_from_idx is not None:
        key ^= z.box_rand[moved_box_from_idx]
    if moved_box_to_idx is not None:
        key ^= z.box_rand[moved_box_to_idx]
    return key
# def highest_bit_index(mask: int) -> int:
#     """
#     Trả về chỉ số bit cao nhất đang bật trong mask.
#     Ví dụ: nếu mask = 0b1001000 → trả về 6.
#     """
#     unsigned = int(mask) if mask >= 0 else mask & 0xFFFFFFFFFFFFFFFF
#     return unsigned.bit_length() - 1 if unsigned != 0 else -1
