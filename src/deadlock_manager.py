# src/deadlock_manager.py
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from .state_codec import BoardIndex, encode_state_from_str

class DeadlockManager:
    """
    Deadlock đơn giản (KHÔNG xét người, KHÔNG dùng JSON):
    - Mỗi deadlock là 1 pattern bitmask các vị trí hộp.
    - Match nhanh: (boxes_mask & pattern) == pattern.
    - Chỉ mục theo 'root' (ô nhỏ nhất trong pattern) + cache theo boxes_mask.
    """
    def __init__(self, board: BoardIndex):
        self.board = board #Tham chiếu cùng board của màn chơi, không copy map
        self._patterns: Set[int] = set() # Tập hợp các pattern deadlock dưới dạng bitmask
        self._by_root: Dict[int, List[int]] = defaultdict(list) # chỉ mục nhanh để giảm duyệt thừa khi match.Gắn mỗi pattern vào 1 root idx nhỏ nhất của patttern đó.
        self._cache_dead: Set[int] = set() #nếu 1 box_mask đã từng match bỏ vào đây
        self._cache_safe: Set[int] = set() # nếu không chứa match pattern nào bỏ vào đây, tăng tốc

    # ---------- tiện ích ----------
    @staticmethod
    #Chuyển danh sách stone thành bitmask
    def _mask_from_seq(stones_seq: List[int]) -> int:
        m = 0
        for s in stones_seq:
            m |= (1 << s)
        return m

    # Xóa cache cũ
    def _invalidate_caches(self) -> None:
        self._cache_dead.clear()
        self._cache_safe.clear()

    # ---------- build API ----------
    #Thêm mẫu vào deadlock từ danh sách stones
    def add_deadlock(self, stones_seq: List[int]) -> None:
        """
        Thêm 1 pattern deadlock (chỉ vị trí hộp).
        stones_seq: danh sách idx ô (vd [16, 21, 29]).
        """
        if not stones_seq:
            return
        mask = self._mask_from_seq(stones_seq)
        if mask in self._patterns:
            return
        self._patterns.add(mask)
        self._by_root[min(stones_seq)].append(mask)
        self._invalidate_caches()
    #Thêm mẫu vào deadlock bằng tọa độ
    def add_deadlock_rc(self, coords: List[Tuple[int, int]]) -> None:
        """
        Thêm pattern theo toạ độ (r,c) thay vì idx.
        """
        stones = []
        for r, c in coords:
            idx = self.board.to_idx[r, c]
            if idx < 0:
                raise ValueError(f"Ô ({r},{c}) không hợp lệ (tường/ngoài map).")
            stones.append(idx)
        self.add_deadlock(stones)

    # ---------- match API ----------
    # Hàm kiểm tra xem boxes_mask có chứa bất kì pattern nào không
    def match_boxes(self, boxes_mask: int) -> bool:
        """True nếu boxes_mask chứa BẤT KỲ pattern deadlock (subset match)."""
        if boxes_mask in self._cache_dead:
            return True
        if boxes_mask in self._cache_safe:
            return False

        bm, i = boxes_mask, 0
        #Lấy bit 1 cuối ra duyệt
        while bm:
            if bm & 1: 
                lst = self._by_root.get(i) # lấy các list cùng 1 box nhỏ 
                if lst:
                    for pat in lst:
                        if (boxes_mask & pat) == pat:
                            self._cache_dead.add(boxes_mask)
                            return True
            bm >>= 1
            i += 1

        self._cache_safe.add(boxes_mask)
        return False
    # DÙng để tra match mà ko có box_match sẵn
    def known_deadlock_from_state(self, state_1line: str) -> bool:
        """Dùng khi bạn chưa có boxes_mask sẵn."""
        boxes_mask, _ = encode_state_from_str(state_1line, self.board)
        return self.match_boxes(boxes_mask) 

    # ---------- quản trị ----------
    def clear(self) -> None:
        """Xoá toàn bộ pattern."""
        self._patterns.clear()
        self._by_root.clear()
        self._invalidate_caches()

    def count_patterns(self) -> int:
        return len(self._patterns)
