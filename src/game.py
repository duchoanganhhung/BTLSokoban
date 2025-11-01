import numpy as np
import pygame

from src.utils import get_state

from .box import Box, Obstacle
from .floor import Floor, Goal
from .player import Player, ReversePlayer


class PuzzleElement:
    def __init__(self, char, obj=None, ground=None):
        self.char = char
        self.ground = ground
        self.obj = obj

    def __str__(self):
        return self.char


class Game:
    def __init__(self, window=None, width=1216, height=640, level=None, seed=None, path=None, panel_cols=0, tile=64):
        self.seed = seed
        self.window = window
        self.level = level

        # Lấy size thật từ cửa sổ
        if self.window is not None:
            width, height = self.window.get_width(), self.window.get_height()

        self.width = width
        self.height = height
        self.panel_cols = int(panel_cols)     # số cột dành cho sidebar (bên phải)
        self.tile = int(tile)                 # kích thước ô (px)

        # Lưới puzzle theo tile hiện tại
        self.puzzle = np.empty((self.height // self.tile, self.width // self.tile), dtype=PuzzleElement)

        self.floor_group = pygame.sprite.Group()
        self.object_group = pygame.sprite.Group()
        self.player_group = pygame.sprite.Group()
        self.goal_group = pygame.sprite.Group()
        self.player = None
        self.puzzle_size = None
        self.pad_x = 0
        self.pad_y = 0
        self.path = path or f'levels/lvl{level}.dat'

        self.load_floor()
        if type(self) == Game:
            self.load_puzzle()

    def __del__(self):
        self.clear_objects()

    # Trả về ma trận ký tự cho A*/BFS (chỉ phần map, không gồm padding)
    def get_matrix(self):
        slice_x = slice(self.pad_x, self.pad_x + self.puzzle_size[1])
        slice_y = slice(self.pad_y, self.pad_y + self.puzzle_size[0])
        sliced = self.puzzle[slice_y, slice_x]
        matrix = np.empty((self.puzzle_size), dtype='<U1')
        for h in range(len(sliced)):
            for w in range(len(sliced[0])):
                matrix[h, w] = sliced[h, w].char
        return matrix

    def get_curr_state(self):
        return get_state(self.get_matrix())

    def print_puzzle(self):
        rows_total = self.height // self.tile
        cols_total = self.width // self.tile
        for h in range(rows_total):
            for w in range(cols_total):
                if self.puzzle[h, w]:
                    print(self.puzzle[h, w].char, end=' ')
                else:
                    print(' ', end=' ')
            print(' ')

    def is_level_complete(self):
        rows_total = self.height // self.tile
        cols_total = self.width // self.tile
        boxes_left = 0
        for h in range(rows_total):
            for w in range(cols_total):
                if self.puzzle[h, w] and self.puzzle[h, w].char == '@':
                    boxes_left += 1
        return boxes_left == 0

    def clear_objects(self):
        for sprite in self.object_group:
            del sprite
        for sprite in self.floor_group:
            del sprite

    def load_floor(self):
        cols_total = self.width  // self.tile
        rows_total = self.height // self.tile
        playable_cols = max(0, cols_total - self.panel_cols)  # chỉ rải đến mép sidebar
        for i in range(playable_cols):
            for j in range(rows_total):
                Floor(self.floor_group, x=i, y=j, tile=self.tile)

    def load_puzzle(self):
        def _tokenize(line: str):
            s = line.strip()
            if not s:
                return []
            # hỗ trợ cả 2 định dạng: có cách và không có cách
            return s.split() if (' ' in s) else list(s)

        try:
            # đọc & chuẩn hoá thành list token theo dòng
            with open(self.path, encoding='utf-8') as f:
                rows_tokens = [_tokenize(ln) for ln in f if _tokenize(ln)]

            if not rows_tokens:
                raise ValueError(f'Empty level file: {self.path}')

            map_cols = len(rows_tokens[0])
            map_rows = len(rows_tokens)
            if any(len(r) != map_cols for r in rows_tokens):
                raise ValueError(f'Inconsistent row length in {self.path}')

            self.puzzle_size = (map_rows, map_cols)

            cols_total = self.width  // self.tile
            rows_total = self.height // self.tile
            playable_cols = max(0, cols_total - self.panel_cols)

            # nếu map lớn hơn vùng playable hoặc cao hơn lưới hiện có => báo lỗi rõ
            if map_cols > playable_cols or map_rows > rows_total:
                raise ValueError(
                    f'Level too big: map {map_rows}x{map_cols} vs grid {rows_total}x{playable_cols} '
                    f'(sidebar {self.panel_cols} cols)'
                )

            # căn giữa TRONG vùng playable (bên trái), không đụng sidebar
            pad_x = max(0, (playable_cols - map_cols) // 2)
            pad_y = max(0, (rows_total - map_rows) // 2)
            self.pad_x, self.pad_y = pad_x, pad_y

            valid = set('+-@X$%* -')

            # tạo phần tử và sprite
            for i, row in enumerate(rows_tokens):
                for j, c in enumerate(row):
                    if len(c) != 1 or c not in valid:
                        raise ValueError(f'Invalid char "{c}" at row {i}, col {j} in {self.path}')

                    new_elem = PuzzleElement(c)
                    self.puzzle[i + pad_y, j + pad_x] = new_elem

                    gx, gy = j + pad_x, i + pad_y  # toạ độ theo lưới

                    if c == '+':  # wall
                        new_elem.obj = Obstacle(self.object_group, x=gx, y=gy, tile=self.tile)
                    elif c == '@':  # box
                        new_elem.obj = Box(self.object_group, x=gx, y=gy, game=self)
                    elif c == '*':  # player
                        new_elem.obj = Player(self.object_group, self.player_group, x=gx, y=gy, game=self)
                        self.player = new_elem.obj
                    elif c == 'X':  # goal
                        new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                    elif c == '$':  # box on goal
                        new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                        new_elem.obj = Box(self.object_group, x=gx, y=gy, game=self)
                    elif c == '%':  # player on goal
                        new_elem.obj = Player(self.object_group, self.player_group, x=gx, y=gy, game=self)
                        new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                        self.player = new_elem.obj
                    # '-' hoặc ' ' => ô trống: không tạo obj/ground

        except (OSError, ValueError) as e:
            import traceback
            traceback.print_exc()
            print(e)
            self.clear_objects()
            return


class ReverseGame(Game):
    def __init__(self, window=None, width=1216, height=640, level=None, seed=None, panel_cols=0, tile=64):
        super().__init__(window, width, height, level, seed, panel_cols=panel_cols, tile=tile)
        self.pad_x = 0
        self.pad_y = 0

    def load_puzzle(self, puzzle):
        # kích thước map đảo
        map_rows = len(puzzle)
        map_cols = len(puzzle[0]) if map_rows > 0 else 0

        cols_total = self.width  // self.tile
        rows_total = self.height // self.tile
        playable_cols = max(0, cols_total - self.panel_cols)

        if map_cols > playable_cols or map_rows > rows_total:
            raise ValueError(
                f'Reverse level too big: map {map_rows}x{map_cols} vs grid {rows_total}x{playable_cols} '
                f'(sidebar {self.panel_cols} cols)'
            )

        # căn giữa trong vùng playable
        pad_x = max(0, (playable_cols - map_cols) // 2)
        pad_y = max(0, (rows_total - map_rows) // 2)
        self.pad_x, self.pad_y = pad_x, pad_y

        for i, row in enumerate(puzzle):
            for j, c in enumerate(row):
                new_elem = PuzzleElement(c)
                self.puzzle[i + pad_y, j + pad_x] = new_elem

                gx, gy = j + pad_x, i + pad_y

                if c == '+':  # wall
                    new_elem.obj = Obstacle(self.object_group, x=gx, y=gy, tile=self.tile)
                elif c == '@':  # box
                    new_elem.obj = Box(self.object_group, x=gx, y=gy, game=self)
                elif c == '*':  # player
                    new_elem.obj = ReversePlayer(self.object_group, self.player_group, x=gx, y=gy, game=self)
                    self.player = new_elem.obj
                elif c == 'X':  # goal
                    new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                elif c == '$':  # box on goal
                    new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                    new_elem.obj = Box(self.object_group, x=gx, y=gy, game=self)
                elif c == '%':  # player on goal
                    new_elem.obj = ReversePlayer(self.object_group, self.player_group, x=gx, y=gy, game=self)
                    new_elem.ground = Goal(self.goal_group, x=gx, y=gy, tile=self.tile)
                    self.player = new_elem.obj
