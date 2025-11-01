import pygame

class Floor(pygame.sprite.Sprite):
    def __init__(self, *groups, x, y, tile=64):
        super().__init__(*groups)
        img = pygame.image.load('img/floor.png').convert_alpha()
        self.image = pygame.transform.scale(img, (tile, tile))
        self.rect = pygame.Rect(x * tile, y * tile, tile, tile)
        self.x = x
        self.y = y
        self.tile = tile

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def __del__(self):
        self.kill()


class Goal(Floor):
    def __init__(self, *groups, x, y, tile=64):
        super().__init__(*groups, x=x, y=y, tile=tile)
        img = pygame.image.load('img/goal.png').convert_alpha()
        self.image = pygame.transform.scale(img, (tile, tile))
        self.rect = pygame.Rect(x * tile, y * tile, tile, tile)
