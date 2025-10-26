import random
from collections import defaultdict

import pygame
from pygame.sprite import Sprite

from .box import Box, Obstacle


class Player(Sprite):
    """A player that can only push boxes"""
    def __init__(self, *groups, x, y, game):
        super().__init__(*groups)
        self.game = game
        t = game.tile

        up = pygame.image.load('img/playerU.png').convert_alpha()
        down = pygame.image.load('img/playerD.png').convert_alpha()
        left = pygame.image.load('img/playerL.png').convert_alpha()
        right = pygame.image.load('img/playerR.png').convert_alpha()

        self.up = pygame.transform.scale(up, (t, t))
        self.down = pygame.transform.scale(down, (t, t))
        self.left = pygame.transform.scale(left, (t, t))
        self.right = pygame.transform.scale(right, (t, t))

        self.image = self.down
        self.rect = pygame.Rect(x * t, y * t, t, t)
        self.x = x   # toạ độ lưới
        self.y = y

    def update(self, key=None):
        # dx, dy theo LƯỚI (ô)
        dx = dy = 0
        if key == 'R':
            self.image = self.right; dx = 1
        elif key == 'L':
            self.image = self.left;  dx = -1
        elif key == 'U':
            self.image = self.up;    dy = -1
        elif key == 'D':
            self.image = self.down;  dy = 1

        if dx == 0 and dy == 0:
            return 0

        curr = (self.y, self.x)
        target = (self.y + dy, self.x + dx)
        target_elem = self.game.puzzle[target]

        # không bước vào obstacle
        if target_elem and target_elem.obj and isinstance(target_elem.obj, Obstacle):
            return 0

        is_box = isinstance(target_elem.obj, Box)
        if (not is_box) or (is_box and target_elem.obj.can_move(dx, dy)):
            curr_elem = self.game.puzzle[curr]
            self.y, self.x = target  # cập nhật toạ độ lưới

            # cập nhật state grid
            curr_elem.char = '-' if not curr_elem.ground else 'X'
            curr_elem.obj = None
            target_elem.char = '*' if not target_elem.ground else '%'
            target_elem.obj = self

            # cập nhật pixel
            t = self.game.tile
            self.rect.topleft = (self.x * t, self.y * t)
            return 1

        return 0

    def __del__(self):
        self.kill()


class ReversePlayer(Player):
    """A player that can only pull boxes"""
    def __init__(self, *groups, x, y, game=None, puzzle=None):
        super().__init__(*groups, x=x, y=y, game=game)
        self.puzzle = puzzle
        self.curr_state = ''
        self.states = defaultdict(int)
        self.prev_move = (0, 0)  # dx, dy theo LƯỚI

    def print_puzzle(self, matrix=None):
        matrix = matrix if matrix is not None else self.game.puzzle
        height, width = len(matrix), len(matrix[0])
        for h in range(height):
            for w in range(width):
                if matrix[h, w]:
                    print(matrix[h, w], end=' ')
                else:
                    print('F', end=' ')
            print(' ')
        print('\n')

    def get_state(self):
        state = ''
        height, width = len(self.game.puzzle), len(self.game.puzzle[0])
        for row in range(height):
            for col in range(width):
                if self.game.puzzle[row, col]:
                    state += str(self.game.puzzle[row, col])
        return state

    def update(self, puzzle_size):
        height, width = puzzle_size
        quick_chars = {
            '*': '-',
            '%': 'X',
            '+': '*',
            '-': '*',
            'X': '%',
            '@': '-',
            '$': 'X',
        }

        # dx, dy theo LƯỚI
        moves_tuples = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        weights = [0.1 if m == self.prev_move else 1 for m in moves_tuples]
        (dx, dy) = random.choices(moves_tuples, weights=weights, k=1)[0]

        self.curr_state = self.get_state()
        self.states[self.curr_state] += 1

        curr_pos = (self.y, self.x)
        target = (self.y + dy, self.x + dx)
        reverse_target = (self.y - dy, self.x - dx)

        # biên padding của map trong lưới
        if (target[1] == self.game.pad_x or
            target[0] == self.game.pad_y or
            target[1] >= self.game.pad_x + width - 1 or
            target[0] >= self.game.pad_y + height - 1 or
            (self.game.puzzle[target] and self.game.puzzle[target].char in '@$')):
            self.prev_move = (dx, dy)
            return

        self.prev_move = (-dx, -dy)

        # cập nhật lưới nhanh
        self.game.puzzle[curr_pos].char = quick_chars[self.game.puzzle[curr_pos].char]
        self.game.puzzle[curr_pos].obj = None
        self.game.puzzle[target].char = quick_chars[self.game.puzzle[target].char]
        if self.game.puzzle[target].obj:
            self.game.puzzle[target].obj.kill()
        self.game.puzzle[target].obj = self

        if (c := self.game.puzzle[reverse_target].char) in '@$':
            self.game.puzzle[reverse_target].char = quick_chars[c]
            self.game.puzzle[reverse_target].obj.reverse_move(dx, dy)

        # cập nhật ảnh & toạ độ pixel
        self.y, self.x = target
        t = self.game.tile
        self.rect.topleft = (self.x * t, self.y * t)
        if (dx, dy) == (1, 0):
            self.image = self.right
        elif (dx, dy) == (-1, 0):
            self.image = self.left
        elif (dx, dy) == (0, 1):
            self.image = self.down
        else:
            self.image = self.up
