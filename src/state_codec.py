"""Unified Zobrist hashing utilities for Sokoban.

This module provides:
- BoardIndex: mapping between (r,c) and compact non-wall indices for bitboards.
- encode_state_from_str: convert flat board string to (boxes_mask, player_idx).
- ZobristHash: a unified class that supports both delta-friendly bitboard hashing
  and full-state/box-only hashing over H*W cells.

Backwards compatibility:
- Zobrist and ZobristHasher are aliases to ZobristHash.
- zobrist_initial and zobrist_update remain available as thin wrappers.
"""

import numpy as np
import random
from typing import Iterable, Optional

WALL = '+'
BOXES = {'@', '$'}
PLAYER = {'*', '%'}

# Biến ma trận 2D thành dạng số hóa 1D bitmask
class BoardIndex:
    """
    Map (r,c) -> idx chỉ số ô hợp lệ (không phải tường) để làm bitboard.
    """
    __slots__ = ('H', 'W', 'to_idx', 'from_idx', 'num')

    def __init__(self, matrix: np.ndarray):
        H, W = matrix.shape
        self.H, self.W = H, W
        self.to_idx = -np.ones((H, W), dtype=int)
        # Ánh xạ tọa độ 2D sang chỉ số 1D bỏ tường, các chỉ số từ 0..n-1 cho các ô đi được
        idx = 0
        for r in range(H):
            for c in range(W):
                if matrix[r, c] != WALL:
                    self.to_idx[r, c] = idx
                    idx += 1
        self.num = idx
        self.from_idx = np.full((idx, 2), -1, dtype=int)
        it = np.where(self.to_idx != -1)
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


class ZobristHash:
    """
    Unified Zobrist hashing supporting two modes:
    - Bitboard mode over non-wall cells (for O(1) delta updates of boxes/player)
    - Grid mode over H*W cells (for full-state hash and order-independent box-only hash)

    Construction options (either or both):
    - num_cells: number of non-wall cells for bitboard hashing.
    - shape: (H, W) for grid hashing.
    """

    # For grid/table hashing
    TILES = ['+', '-', 'X', '*', '%', '@', '$']
    TILE_INDEX = {c: i for i, c in enumerate(TILES)}

    def __init__(
        self,
        num_cells: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        *,
        seed_bitboard: int = 1337,
        seed_grid: int = 0xC0FFEE,
    ):
        # Bitboard tables (only if num_cells provided)
        self.box_rand = None
        self.player_rand = None
        if num_cells is not None:
            rnd = random.Random(seed_bitboard)
            self.box_rand = [rnd.getrandbits(64) for _ in range(int(num_cells))]
            self.player_rand = [rnd.getrandbits(64) for _ in range(int(num_cells))]

        # Grid tables (only if shape provided)
        self.h = None
        self.w = None
        self.table = None
        self.box_table = None
        if shape is not None:
            h, w = shape
            self.h = int(h)
            self.w = int(w)
            rnd = random.Random(seed_grid)
            self.table = [
                [rnd.getrandbits(64) for _ in range(len(self.TILES))]
                for _ in range(self.h * self.w)
            ]
            # Per-position randoms for box-only hashing (order-independent XOR)
            self.box_table = [rnd.getrandbits(64) for _ in range(self.h * self.w)]

    # -------- Bitboard (boxes_mask, player_idx) helpers --------
    def initial_key(self, boxes_mask: int, player_idx: int) -> int:
        """Compute initial Zobrist key from bitboard data.
        Requires bitboard tables to be initialized (num_cells passed to ctor)."""
        if self.box_rand is None or self.player_rand is None:
            raise RuntimeError("Bitboard tables not initialized (num_cells not provided)")
        key = 0
        bm = boxes_mask
        idx = 0
        while bm:
            if bm & 1:
                key ^= self.box_rand[idx]
            bm >>= 1
            idx += 1
        if player_idx >= 0:
            key ^= self.player_rand[player_idx]
        return key

    def update_key(
        self,
        key: int,
        old_player_idx: int,
        new_player_idx: int,
        moved_box_from_idx: int | None,
        moved_box_to_idx: int | None,
    ) -> int:
        """Delta-update an existing bitboard key in O(1)."""
        if self.box_rand is None or self.player_rand is None:
            raise RuntimeError("Bitboard tables not initialized (num_cells not provided)")
        # player XOR out/in
        if old_player_idx >= 0:
            key ^= self.player_rand[old_player_idx]
        if new_player_idx >= 0:
            key ^= self.player_rand[new_player_idx]
        # box XOR out/in nếu có đẩy hộp
        if moved_box_from_idx is not None:
            key ^= self.box_rand[moved_box_from_idx]
        if moved_box_to_idx is not None:
            key ^= self.box_rand[moved_box_to_idx]
        return key

    # -------- Grid (H*W) helpers --------
    def hash_state(self, state: str) -> int:
        """Hash entire H*W state string (includes walls, floors, player, boxes)."""
        if self.table is None:
            raise RuntimeError("Grid tables not initialized (shape not provided)")
        hval = 0
        for idx, ch in enumerate(state):
            t = self.TILE_INDEX.get(ch)
            if t is None:
                # ignore unknown tiles (shouldn't happen in normal puzzles)
                continue
            hval ^= self.table[idx][t]
        return hval

    def hash_boxes(self, state: str) -> int:
        """Order-independent hash for box-only configuration (ignores player).
        Treats both '@' and '$' as boxes so on-goal/off-goal differences are ignored,
        matching existing boxes_signature behavior."""
        if self.box_table is None:
            raise RuntimeError("Grid tables not initialized (shape not provided)")
        hval = 0
        for idx, ch in enumerate(state):
            if ch == '@' or ch == '$':
                hval ^= self.box_table[idx]
        return hval


# # -------- Backwards-compatible helpers/aliases --------
# def zobrist_initial(z: "ZobristHash", boxes_mask: int, player_idx: int) -> int:
#     return z.initial_key(boxes_mask, player_idx)


# def zobrist_update(
#     z: "ZobristHash",
#     key: int,
#     old_player_idx: int,
#     new_player_idx: int,
#     moved_box_from_idx: int | None,
#     moved_box_to_idx: int | None,
# ) -> int:
#     return z.update_key(key, old_player_idx, new_player_idx, moved_box_from_idx, moved_box_to_idx)


# # Aliases so existing imports keep working
# Zobrist = ZobristHash
# ZobristHasher = ZobristHash
