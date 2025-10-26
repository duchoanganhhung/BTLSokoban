# src/deadlock_build_simple.py
from itertools import combinations
from typing import List, Dict, Set, Optional

from .state_codec import BoardIndex
from .deadlock_manager import DeadlockManager

BitMask = int

# Tìm các điểm kề của 1 ô hợp lệ trong board và trả về danh sách là các board_idx
def _build_neighbors(board: BoardIndex) -> List[List[int]]:
    """Danh sách kề 4 hướng cho mọi ô hợp lệ (không phải tường)."""
    nei = [[] for _ in range(board.num)]
    for i, (r, c) in enumerate(board.from_idx):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < board.H and 0 <= nc < board.W:
                j = board.to_idx[nr, nc]
                if j >= 0:
                    nei[i].append(j)
    return nei
#Máy tạo deadlock tự động cho màn chơi, sau đó chuyển từng deadlock vào manager quản lí
class RetroBuilderSimple:
    """
    Retrograde 'đơn giản – không xét người':
      - State = bitmask hộp (không có vị trí người).
      - Một 'push' hợp lệ: d (ô trước mặt) trống, p (ô lùi) trống.
        (Không kiểm tra người tiếp cận p: chấp nhận xấp xỉ 'đơn giản'.)
      - Deadlock k-hộp: mọi push đều dẫn đến deadlock đã biết (cùng k).
    Có 3 thủ tục như Mục 4: combinations / subsequent / smaller-deadlocks (Pmax).
    Kết quả -> nạp thẳng vào DeadlockManager (pattern-only).
    """
    def __init__(self,
                 board: BoardIndex,
                 deadlock_mgr: DeadlockManager,
                 cells: Optional[List[int]] = None,
                 matrix = None):
        self.board = board 
        self.dlm = deadlock_mgr
        self.nei = _build_neighbors(board) #danh sách các nei của từng ô hợp lệ
        # Miền ô xét đặt hộp: mặc định tất cả ô hợp lệ
        self.cells = cells if cells is not None else list(range(board.num))
        # Bảng deadlock theo số hộp
        self.deadlocks_by_size: Dict[int, Set[BitMask]] = {}
        if matrix is not None:
            gm = 0 
            for i,(r,c) in enumerate(board.from_idx):
                if matrix[r, c] in ('X', '$', '%'):
                   gm |= (1 << i)
            self.goals_mask = gm
        else:
            self.goals_mask = 0

    # ---------- bit utils ----------
    @staticmethod
    def _has(m: BitMask, i: int) -> bool:
        m = int(m)
        i = int(i)
        #print("Kieu dữ liệu trong has",type(m),type(i))
        return (m >> i) & 1 # ô i có hộp hay k 
    @staticmethod
    def _set(m: BitMask, i: int) -> BitMask: 
        m = int(m)
        i = int(i)
        return m | (1 << i) # set hộp ở ô i là 1
    @staticmethod
    def _clr(m: BitMask, i: int) -> BitMask: 
        m = int(m)
        i = int(i)
        return m & ~(1 << i) # bỏ hộp ở ô i là 1

    # ---------- sinh các 'push' hợp lệ từ state ----------# Nhằm check xem tất cả nước đi này đều quy về deadlock đã biết
    # sinh theo chiều thuận, từ state này sinh ra chứ ko phải từ deadlock đi ngược lại
    def _legal_push_children(self, stones: BitMask) -> List[BitMask]:
        """
        Với mỗi hộp s và mỗi hướng đến d:
          - d phải trống, p (ô sau s theo hướng ngược) phải trống.
          - Không xét người (đơn giản): chỉ kiểm free/biên.
        """
        outs: List[BitMask] = [] #khởi tạo danh sách kết quả
        for s in self.cells:
            if not self._has(stones, s):
                continue
            sy, sx = self.board.from_idx[s]
            for d in self.nei[s]:
                if self._has(stones, d):  # đích có hộp
                    continue
                dy, dx = self.board.from_idx[d][0] - sy, self.board.from_idx[d][1] - sx
                py, px = sy - dy, sx - dx # tọa độ ô lùi, đảm bảo có người đứng đẩy được
                p = self.board.to_idx[py, px] if (0 <= py < self.board.H and 0 <= px < self.board.W) else -1
                if p < 0 or self._has(stones, p):  # lùi không hợp lệ
                    continue
                child = self._clr(stones, s) #bỏ hộp ở trạng thái hiện tại
                child = self._set(child, d) # thêm hộp vào ô mới
                outs.append(child)
        return outs

    # ---------- kiểm tra 'mọi push từ parent đều vào known' ----------
    def _all_moves_to_known(self, parent: BitMask, known: Set[BitMask]) -> bool:
        if (parent & ~self.goals_mask) == 0:
            return False
        kids = self._legal_push_children(parent)
        if not kids:
            return True  # không có nước đẩy nào -> xem là deadlock
        for ch in kids:
            if ch not in known:
                return False
        return True

    # ---------- FIRST PASS: combinations (k hộp) ----------
    # Bước đầu tiên tạo tất cả các tổ hợp deadlock có thể của k hộp
    # Tạo bằng cách dẫn nó tới deadlock đã biết hoặc ko đẩy được nữa
    # Có thể có cha hoặc cha cha cha ... của nhiều cái , tuy nhiên có thể vẫn chưa đủ vì lúc xét known thì con có thể vẫn chưa được tìm thấy
    def combinations_deadlocks_search(self, k: int) -> List[BitMask]:
        res: List[BitMask] = [] # danh sách kết quả trả về deadlock
        known = self.deadlocks_by_size.setdefault(k, set()) # tập deadlock đã biết sau khi làm
        # Duyệt mọi tổ hợp k ô trong 'cells'
        for comb in combinations(self.cells, k):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            if mask in known:
                continue
            if self._all_moves_to_known(mask, known):
                known.add(mask)
                res.append(mask)
        return res

    # ---------- SUBSEQUENT PASSES ----------
    def subsequent_deadlocks_search(self, k: int, seeds: List[BitMask]) -> List[BitMask]:
        res: List[BitMask] = []
        known = self.deadlocks_by_size.setdefault(k, set())
        for st in seeds:
            # tạo 'parent' bằng kéo ngược: d (hiện tại) -> p (parent)
            for d in self.cells:
                if not self._has(st, d):
                    continue
                dy, dx = self.board.from_idx[d]
                for p in self.nei[d]:
                    if self._has(st, p):
                        continue
                    py, px = self.board.from_idx[p]
                    v_y, v_x = dy - py, dx - px           # vector p->d
                    by, bx = py - v_y, px - v_x           # ô 'sau lưng' p theo hướng đẩy
                    b = self.board.to_idx[by, bx] if (0 <= by < self.board.H and 0 <= bx < self.board.W) else -1
                    if b < 0:
                        continue
                    parent = self._clr(st, d)
                    parent = self._set(parent, p)
                    # điều kiện đẩy từ parent p->d: b & d đều phải trống trong parent
                    if self._has(parent, b) or self._has(parent, d):
                        continue
                    if parent in known:
                        continue
                    if self._all_moves_to_known(parent, known):
                        known.add(parent)
                        res.append(parent)
        return res

    # ---------- FIRST PASS bằng 'smaller deadlocks' (k lớn) ----------
    def smaller_deadlocks_search(self, k: int, Pmax: int = 2) -> List[BitMask]:
        res: List[BitMask] = []
        known = self.deadlocks_by_size.setdefault(k, set())
        for P in range(1, min(Pmax, k - 1) + 1):
            base_size = k - P
            base_list = list(self.deadlocks_by_size.get(base_size, set()))
            if not base_list:
                continue
            # Lấy parent của base (kéo ngược 1 bước), rồi cộng thêm P ô trống bất kỳ
            foundation: set[int] = set(base_list)
            for st in base_list:
            # thu thập TẤT CẢ parent 1 bước của st (KHÔNG lọc theo deadlock)
              for d in self.cells:
                if not self._has(st, d):
                    continue  # d là vị trí đang có hộp trong st
                dy, dx = self.board.from_idx[d]
                for p in self.nei[d]:
                    if self._has(st, p):
                        continue  # p phải trống trong st
                    py, px = self.board.from_idx[p]
                    v_y, v_x = dy - py, dx - px       # p->d
                    by, bx = py - v_y, px - v_x       # ô lùi sau p
                    if not (0 <= by < self.board.H and 0 <= bx < self.board.W):
                        continue
                    b = self.board.to_idx[by, bx]
                    if b < 0:
                        continue

                    parent = self._clr(st, d)
                    parent = self._set(parent, p)

                    # push thuận từ parent: b & d phải TRỐNG trong parent
                    if self._has(parent, b) or self._has(parent, d):
                        continue

                    # (tuỳ chọn) giữ trong miền cells:
                    # if p not in self.cells: 
                    #     continue

                    foundation.add(parent)          

            for parent in foundation:
                # các ô trống trong parent
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

    # ---------- Orchestrator: giống Mục 4 ----------
    def build(self, max_k: int, max_comb_k: int = 4, Pmax: int = 2) -> None:
        print ("Đang bắt đầu build deadlock\n")
        for k in range(1, max_k + 1):
            if k <= max_comb_k:
                news = self.combinations_deadlocks_search(k)
            else:
                news = self.smaller_deadlocks_search(k, Pmax=Pmax)
            while news:
                news = self.subsequent_deadlocks_search(k, news)

        # Đổ toàn bộ pattern vào DeadlockManager (pattern-only)
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
        # print("Kết thúc build deadlock\n")
        # for k in sorted(self.deadlocks_by_size.keys()):
        #     masks = sorted(int(m) for m in self.deadlocks_by_size[k])
        #     print(f"\n=== DEADLOCK k ={k} (count= {len(masks)}) ===")
        #     for m in masks:
        #         print(f"mask={m} (0x{m:x})")
        #         self._print_mask_matrix(m)
        #         print()
    def _print_mask_matrix(self, mask: int) -> None:
        """In ma trận với: '+' tường, '-' sàn, 'X' goal, 'B' box, '*' box-on-goal."""
        H, W = self.board.H, self.board.W
        grid = [['+' if self.board.to_idx[r, c] < 0 else '-' for c in range(W)] for r in range(H)]

        # vẽ goal trước
        gm = int(self.goals_mask)
        for i in range(self.board.num):
            if (gm >> i) & 1:
                r, c = map(int, self.board.from_idx[i])
                grid[r][c] = 'X'

        # overlay box / box-on-goal
        m = int(mask)
        for i in range(self.board.num):
            if (m >> i) & 1:
                r, c = map(int, self.board.from_idx[i])
                grid[r][c] = '*' if ((gm >> i) & 1) else 'B'

        print("\n".join("".join(row) for row in grid))    
