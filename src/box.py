import pygame
from pygame.sprite import Sprite

class Box(Sprite):
    def __init__(self, *groups, x, y, game=None):
        super().__init__(*groups)
        self.game = game
        t = game.tile if game is not None else 64

        img = pygame.image.load('img/box.png').convert_alpha()
        self.sprite = pygame.transform.scale(img, (t, t))

        imgg = pygame.image.load('img/boxg.png').convert_alpha()
        self.spriteg = pygame.transform.scale(imgg, (t, t))

        # chọn sprite theo việc đứng trên goal hay không
        self.image = self.sprite if (game and not game.puzzle[y, x].ground) else self.spriteg
        self.rect = pygame.Rect(x * t, y * t, t, t)

        self.x = x  # toạ độ lưới
        self.y = y
        self.tile = t

    def can_move(self, dx, dy):
        """
        dx, dy: delta THEO LƯỚI (ô), không phải pixel.
        Chỉ cho di chuyển nếu ô đích trống (không có obj).
        """
        target_x, target_y = self.x + dx, self.y + dy
        target = (target_y, target_x)
        curr   = (self.y, self.x)

        target_elem = self.game.puzzle[target]
        if target_elem.obj is None:  # ô trống (có thể có ground)
            curr_elem = self.game.puzzle[curr]

            # cập nhật vị trí box trong lưới
            self.x, self.y = target_x, target_y

            # cập nhật state grid
            curr_elem.char = '-' if not curr_elem.ground else 'X'
            curr_elem.obj = None
            target_elem.char = '@' if not target_elem.ground else '$'
            target_elem.obj = self

            # cập nhật toạ độ pixel
            self.rect.topleft = (self.x * self.tile, self.y * self.tile)
            self.update_sprite()
            return True
        return False

    def reverse_move(self, dx, dy):
        """Di chuyển ngược (dx,dy theo lưới)."""
        target_y, target_x = self.y + dx, self.x + dy
        curr_pos = (self.y, self.x)

        self.game.puzzle[curr_pos].obj = None
        self.game.puzzle[(target_y, target_x)].obj = self

        self.y, self.x = target_y, target_x
        self.rect.topleft = (self.x * self.tile, self.y * self.tile)

        self.game.puzzle[curr_pos].char = 'X' if self.game.puzzle[curr_pos].ground else '-'
        self.game.puzzle[(target_y, target_x)].char = '$' if self.game.puzzle[(target_y, target_x)].ground else '@'
        self.update_sprite()

    def update_sprite(self):
        curr_obj = self.game.puzzle[self.y, self.x]
        self.image = self.spriteg if curr_obj and curr_obj.ground else self.sprite

    def __del__(self):
        self.kill()


class Obstacle(Sprite):
    """Tường/cột – vật cản tĩnh, không đẩy được."""
    def __init__(self, *groups, x, y, tile=64):
        super().__init__(*groups)
        img = pygame.image.load('img/obs.png').convert_alpha()
        self.image = pygame.transform.scale(img, (tile, tile))
        self.rect = pygame.Rect(x * tile, y * tile, tile, tile)
        self.x = x
        self.y = y
        self.tile = tile
