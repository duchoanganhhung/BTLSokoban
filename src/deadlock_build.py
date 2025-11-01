# src/deadlock_build_simple.py
from itertools import combinations
from typing import List, Dict, Set, Optional

from .state_codec import BoardIndex
from .deadlock_manager import DeadlockManager

BitMask = int  # dùng int để biểu diễn bitmask của toàn bộ vị trí hộp


# ----------------------------
# Hàm tạo danh sách các ô kề nhau cho mỗi ô hợp lệ trong board (không phải tường)
# ----------------------------
def _build_neighbors(board: BoardIndex) -> List[List[int]]:
    nei = [[] for _ in range(board.num)]
    for i, (r, c) in enumerate(board.from_idx):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            # Nếu ô kề hợp lệ (không vượt ra khỏi board và không là tường)
            if 0 <= nr < board.H and 0 <= nc < board.W:
                j = board.to_idx[nr, nc]
                if j >= 0:  # chỉ lấy ô hợp lệ (không phải tường)
                    nei[i].append(j)
    return nei


# ----------------------------
# Class chính xây dựng danh sách DEADLOCK theo hướng retrograde (ngược)
# Không xét đến vị trí người, chỉ xét vị trí hộp (đơn giản hóa)
# ----------------------------
class RetroBuilderSimple:
    def __init__(self, board: BoardIndex, deadlock_mgr: DeadlockManager,
                 cells: Optional[List[int]] = None, matrix=None):
        self.board = board  # bảng chứa ánh xạ (r,c) <-> idx
        self.dlm = deadlock_mgr  # đối tượng quản lý deadlock
        self.nei = _build_neighbors(board)  # danh sách kề của từng ô hợp lệ
        # các ô hợp lệ có thể đặt hộp
        self.cells = cells if cells is not None else list(range(board.num))
        self.deadlocks_by_size: Dict[int, Set[BitMask]] = {}  # lưu các deadlock phân theo số hộp

        # Xây goal mask (bitmask các ô goal)
        if matrix is not None:
            gm = 0
            for i, (r, c) in enumerate(board.from_idx):
                if matrix[r, c] in ('X', '$', '%'):
                    gm |= (1 << i)
            self.goals_mask = gm
        else:
            self.goals_mask = 0

    # --------------------------
    # Các hàm thao tác bitmask
    # --------------------------
    @staticmethod
    def _has(m: BitMask, i: int) -> bool:
        """Kiểm tra ô i có hộp không."""
        m = int(m)
        i = int(i)
        return (m >> i) & 1

    @staticmethod
    def _set(m: BitMask, i: int) -> BitMask:
        """Đặt hộp tại ô i (bit = 1)."""
        m = int(m)
        i = int(i)
        return m | (1 << i)

    @staticmethod
    def _clr(m: BitMask, i: int) -> BitMask:
        """Xóa hộp tại ô i (bit = 0)."""
        m = int(m)
        i = int(i)
        return m & ~(1 << i)

    # --------------------------
    # Sinh tất cả nước đẩy hợp lệ từ một state
    # (đơn giản, không xét người)
    # --------------------------
    def _legal_push_children(self, stones: BitMask) -> List[BitMask]:
        outs: List[BitMask] = []
        for s in self.cells:
            if not self._has(stones, s):
                continue
            sy, sx = self.board.from_idx[s]
            # duyệt tất cả hướng có thể đẩy
            for d in self.nei[s]:
                if self._has(stones, d):  # ô đích có hộp => không hợp lệ
                    continue
                dy, dx = self.board.from_idx[d][0] - sy, self.board.from_idx[d][1] - sx
                py, px = sy - dy, sx - dx  # ô lùi lại (vị trí người đẩy)
                # kiểm tra hợp lệ
                p = self.board.to_idx[py, px] if (0 <= py < self.board.H and 0 <= px < self.board.W) else -1
                if p < 0 or self._has(stones, p):
                    continue
                # sinh trạng thái mới
                child = self._clr(stones, s)
                child = self._set(child, d)
                outs.append(child)
        return outs

    # --------------------------
    # Kiểm tra xem mọi nước đẩy từ parent đều dẫn đến deadlock đã biết chưa
    # --------------------------
    def _all_moves_to_known(self, parent: BitMask, known: Set[BitMask]) -> bool:
        # Nếu tất cả hộp đều nằm trên goal thì KHÔNG coi là deadlock
        if (parent & ~self.goals_mask) == 0:
            return False
        # Sinh các state con từ parent
        kids = self._legal_push_children(parent)
        # Nếu không có con hợp lệ => coi là deadlock
        if not kids:
            return True
        # Nếu có con nhưng có con nào chưa trong known => chưa chắc deadlock
        for ch in kids:
            if ch not in known:
                return False
        return True

    # --------------------------
    # FIRST PASS: sinh deadlock bằng cách duyệt tổ hợp các ô chứa hộp
    # --------------------------
    def combinations_deadlocks_search(self, k: int) -> List[BitMask]:
        res: List[BitMask] = []
        known = self.deadlocks_by_size.setdefault(k, set())
        # Duyệt mọi tổ hợp k ô hợp lệ
        for comb in combinations(self.cells, k):
            # Bỏ qua tổ hợp có hộp trong góc chết (3 tường bao quanh)
            skip = False
            for idx in comb:
                r, c = self.board.from_idx[idx]
                wall_count = 0
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board.H and 0 <= nc < self.board.W:
                        if self.board.to_idx[nr, nc] == -1:
                            wall_count += 1
                if wall_count >= 3:  # nếu ô bị kẹp bởi >=3 tường
                    skip = True
                    break
            if skip:
                continue

            # Biểu diễn tổ hợp này thành bitmask
            mask = 0
            for i in comb:
                mask |= (1 << i)

            # Nếu đã biết rồi thì bỏ qua
            if mask in known:
                continue
            # Nếu tất cả nước đi con đều thuộc deadlock đã biết
            if self._all_moves_to_known(mask, known):
                known.add(mask)
                res.append(mask)
        return res

    # --------------------------
    # SUBSEQUENT PASS: sinh thêm deadlock từ các seed đã có (kéo ngược)
    # --------------------------
    def subsequent_deadlocks_search(self, k: int, seeds: List[BitMask]) -> List[BitMask]:
        res: List[BitMask] = []
        known = self.deadlocks_by_size.setdefault(k, set())
        for st in seeds:
            for d in self.cells:
                if not self._has(st, d):
                    continue
                dy, dx = self.board.from_idx[d]
                for p in self.nei[d]:
                    if self._has(st, p):
                        continue
                    py, px = self.board.from_idx[p]
                    v_y, v_x = dy - py, dx - px
                    by, bx = py - v_y, px - v_x
                    b = self.board.to_idx[by, bx] if (0 <= by < self.board.H and 0 <= bx < self.board.W) else -1
                    if b < 0:
                        continue
                    parent = self._clr(st, d)
                    parent = self._set(parent, p)
                    if self._has(parent, b) or self._has(parent, d):
                        continue
                    if parent in known:
                        continue
                    if self._all_moves_to_known(parent, known):
                        known.add(parent)
                        res.append(parent)
        return res

    # --------------------------
    # SMALLER DEADLOCK PASS:
    # tạo deadlock lớn hơn từ deadlock nhỏ (thêm P ô trống bất kỳ)
    # --------------------------
    def smaller_deadlocks_search(self, k: int, Pmax: int = 2) -> List[BitMask]:
        res: List[BitMask] = []
        known = self.deadlocks_by_size.setdefault(k, set())
        for P in range(1, min(Pmax, k - 1) + 1):
            base_size = k - P
            base_list = list(self.deadlocks_by_size.get(base_size, set()))
            if not base_list:
                continue
            foundation: set[int] = set(base_list)
            # Sinh tất cả parent 1 bước từ base
            for st in base_list:
                for d in self.cells:
                    if not self._has(st, d):
                        continue
                    dy, dx = self.board.from_idx[d]
                    for p in self.nei[d]:
                        if self._has(st, p):
                            continue
                        py, px = self.board.from_idx[p]
                        v_y, v_x = dy - py, dx - px
                        by, bx = py - v_y, px - v_x
                        if not (0 <= by < self.board.H and 0 <= bx < self.board.W):
                            continue
                        b = self.board.to_idx[by, bx]
                        if b < 0:
                            continue
                        parent = self._clr(st, d)
                        parent = self._set(parent, p)
                        if self._has(parent, b) or self._has(parent, d):
                            continue
                        foundation.add(parent)

            # Thêm P ô trống bất kỳ để tạo deadlock lớn hơn
            for parent in foundation:
                free_cells = [i for i in self.cells if not self._has(parent, i)]
                for extra in combinations(free_cells, P):
                    cand = parent
                    ok = True
                    for e in extra:
                        if self._has(cand, e):
                            ok = False; break
                        cand = self._set(cand, e)
                    if not ok or cand in known:
                        continue
                    if self._all_moves_to_known(cand, known):
                        known.add(cand)
                        res.append(cand)
        return res

    # --------------------------
    # HÀM CHÍNH ĐIỀU PHỐI BUILD DEADLOCK
    # --------------------------
    def build(self, max_k: int, max_comb_k: int = 4, Pmax: int = 2) -> None:
        print("\n[DEADLOCK] Building...")
        for k in range(1, max_k + 1):
            if k <= max_comb_k:
                news = self.combinations_deadlocks_search(k)
            else:
                news = self.smaller_deadlocks_search(k, Pmax=Pmax)
            while news:
                news = self.subsequent_deadlocks_search(k, news)

        # Thêm toàn bộ deadlock vào DeadlockManager
        for k, masks in self.deadlocks_by_size.items():
            for m in masks:
                stones_seq = []
                mm, idx = m, 0
                while mm:
                    if mm & 1:
                        stones_seq.append(idx)
                    mm >>= 1; idx += 1
                if stones_seq:
                    stones_seq.sort()
                    self.dlm.add_deadlock(stones_seq)
        print("[DEADLOCK] Done.")
