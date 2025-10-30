import random
from typing import Iterable


class ZobristHasher:
    """
    Lightweight Zobrist hashing for Sokoban states.
    - State is a flat string of length H*W with characters in {'+','-','X','*','%','@','$'}
    - Hash is XOR of random 64-bit numbers per (cell, tileType)
    Deterministic given default seed for reproducibility in tests.
    """

    TILES = ['+', '-', 'X', '*', '%', '@', '$']
    TILE_INDEX = {c: i for i, c in enumerate(TILES)}

    def __init__(self, shape: Iterable[int], seed: int = 0xC0FFEE):
        h, w = shape
        self.h = h
        self.w = w
        rnd = random.Random(seed)
        self.table = [
            [rnd.getrandbits(64) for _ in range(len(self.TILES))]
            for _ in range(h * w)
        ]
        # Per-position randoms for box-only hashing (order-independent XOR)
        self.box_table = [rnd.getrandbits(64) for _ in range(h * w)]

    def hash_state(self, state: str) -> int:
        h = 0
        for idx, ch in enumerate(state):
            t = self.TILE_INDEX.get(ch)
            if t is None:
                # ignore unknown tiles (shouldn't happen in normal puzzles)
                continue
            h ^= self.table[idx][t]
        return h

    def hash_boxes(self, state: str) -> int:
        """Order-independent hash for box-only configuration (ignores player).
        Treats both '@' and '$' as boxes so on-goal/off-goal differences are ignored,
        matching existing boxes_signature behavior."""
        h = 0
        for idx, ch in enumerate(state):
            if ch == '@' or ch == '$':
                h ^= self.box_table[idx]
        return h
