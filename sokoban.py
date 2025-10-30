import random
import time
import os
import re

import pygame
import pygame_widgets

from src.astar import solve_astar
from src.bfs import solve_bfs
from src.events import *
from src.game import Game
# generator-based random puzzles removed: no import
from src.utils import play_solution
from src.widgets import sidebar_widgets

# --- cấu hình khung và panel ---
BASE_W, BASE_H = 1216, 640
SIDEBAR_COLS = 4       # số cột ô dành riêng cho panel (bên phải)
MAX_TILE = 64          # ô lớn nhất
MIN_TILE = 24          # ô nhỏ nhất để vẫn nhìn rõ


def get_max_level(levels_dir='levels'):
    """
    Trả về số cấp độ lớn nhất tìm thấy trong các tệp có tên `lvlN.dat` trong levels_dir.
    Nếu không tìm thấy, trả về 1.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    levels_path = levels_dir if os.path.isabs(levels_dir) else os.path.join(base_dir, levels_dir)

    try:
        files = os.listdir(levels_path)
    except Exception:
        return 1
    nums = []
    for f in files:
        m = re.match(r'lvl(\d+)\.dat$', f)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                continue
    return max(nums) if nums else 1


MAX_LEVEL = get_max_level()

random.seed(6)

# Đo kích thước level từ file .dat (theo số token cách nhau bởi dấu cách)
def probe_level_size(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rows = len(lines)
    cols = len(lines[0].split()) if rows > 0 else 0
    return rows, cols

# Chọn kích thước ô sao cho map + sidebar lọt trong cửa sổ cố định
def compute_tile(window_w, window_h, cols, rows, sidebar_cols=4, max_tile=64, min_tile=24):
    # (cols + sidebar_cols) * tile <= window_w  và  rows * tile <= window_h
    t_fit_w = window_w // (cols + sidebar_cols) if (cols + sidebar_cols) > 0 else max_tile
    t_fit_h = window_h // rows if rows > 0 else max_tile
    t = min(max_tile, t_fit_w, t_fit_h)
    return max(min_tile, t)

# ---------------- Vòng đời một màn chơi ----------------
def play_game(window, level=1, random_game=False, random_seed=None, tile=64, **widgets):
    # Thiết lập trạng thái
    moves = 0
    runtime = 0
    show_solution = False
    widgets['paths'].transparency = False

    # random-game feature removed: always load explicit levels

    if level <= 1:
        widgets['prev_button'].hide()
    else:
        widgets['prev_button'].show()

    if level >= MAX_LEVEL:
        widgets['next_button'].hide()
    else:
        widgets['next_button'].show()

    widgets['label'].set_text(f'Level {level}', 30)

    # Tạo game với tile động + chừa sidebar
    game = Game(level=level, window=window, panel_cols=SIDEBAR_COLS, tile=int(tile))
    game_loop = True

    while game_loop:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                game_loop = False
                return {'keep_playing': False, 'reset': -1}

            elif event.type == RESTART_EVENT:
                game_loop = False
                print(f'Restarting level {level}\n')
                window.fill((0, 0, 0, 0))
                return {'keep_playing': True, 'reset': level}

            elif event.type == PREVIOUS_EVENT:
                game_loop = False
                print(f'Previous level {level - 1}\n')
                window.fill((0, 0, 0, 0))
                return {'keep_playing': True, 'reset': level - 1}

            elif event.type == NEXT_EVENT:
                game_loop = False
                print(f'Next level {level + 1}\n')
                window.fill((0, 0, 0, 0))
                return {'keep_playing': True, 'reset': level + 1}

            # RANDOM_GAME_EVENT handling removed — random puzzles disabled

            elif event.type == SOLVE_BFS_EVENT:
                print('Finding a solution for the puzzle\n')
                widgets['paths'].reset('Solving with [BFS]')
                show_solution = True
                start = time.time()
                solution, depth = solve_bfs(
                    game.get_matrix(),
                    widget=widgets['paths'],
                    visualizer=widgets['toggle'].getValue()
                )
                runtime = round(time.time() - start, 5)
                if solution:
                    widgets['paths'].solved = True
                    widgets['paths'].transparency = True
                    widgets['paths'].set_text(f'[BFS] Solution Found in {runtime}s!\n{solution}', 20)
                    moves = play_solution(solution, game, widgets, show_solution, moves)
                else:
                    widgets['paths'].solved = False
                    widgets['paths'].set_text(
                        '[BFS] Solution Not Found!\n' + ('Deadlock Found!' if depth < 0 else f'Depth {depth}'),
                        20
                    )

            elif event.type == SOLVE_ASTARMAN_EVENT:
                print('Finding a solution for the puzzle\n')
                widgets['paths'].reset('Solving with [A*]')
                show_solution = True
                start = time.time()
                solution, depth = solve_astar(
                    game.get_matrix(),
                    widget=widgets['paths'],
                    visualizer=widgets['toggle'].getValue(),
                    heuristic='manhattan',
                    macro=True
                )
                runtime = round(time.time() - start, 5)
                if solution:
                    widgets['paths'].solved = True
                    widgets['paths'].transparency = True
                    widgets['paths'].set_text(f'[A*] Solution Found in {runtime}s!\n{solution}', 20)
                    moves = play_solution(solution, game, widgets, show_solution, moves)
                else:
                    widgets['paths'].solved = False
                    widgets['paths'].set_text(
                        '[A*] Solution Not Found!\n' + ('Deadlock Found!' if depth < 0 else f'Depth {depth}'),
                        20
                    )

            elif event.type == SOLVE_DIJKSTRA_EVENT:
                print('Finding a solution for the puzzle\n')
                widgets['paths'].reset('Solving with [Dijkstra]')
                show_solution = True
                start = time.time()
                solution, depth = solve_astar(
                    game.get_matrix(),
                    widget=widgets['paths'],
                    visualizer=widgets['toggle'].getValue(),
                    heuristic='dijkstra',
                )
                runtime = round(time.time() - start, 5)
                if solution:
                    widgets['paths'].solved = True
                    widgets['paths'].transparency = True
                    widgets['paths'].set_text(f'[Dijkstra] Solution Found in {runtime}s!\n{solution}', 20)
                    moves = play_solution(solution, game, widgets, show_solution, moves)
                else:
                    widgets['paths'].solved = False
                    widgets['paths'].set_text(
                        '[Dijkstra] Solution Not Found!\n' + ('Deadlock Found!' if depth < 0 else f'Depth {depth}'),
                        20
                    )

            elif event.type == SOLVE_HUNGARIAN_EVENT:
                print('Finding a solution for the puzzle\n')
                widgets['paths'].reset('Solving with [Hungarian]')
                show_solution = True
                start = time.time()
                solution, depth = solve_astar(
                    game.get_matrix(), 
                    widget=widgets['paths'], 
                    visualizer=widgets['toggle'].getValue(),
                    heuristic='hungarian',
                )
                runtime = round(time.time() - start, 5)
                if solution:
                    widgets['paths'].solved = True
                    widgets['paths'].transparency = True
                    widgets['paths'].set_text(
                        f'[Hungarian] Solution Found in {runtime}s!\n{solution}',
                        20
                    )
                    moves = play_solution(solution, game, widgets, show_solution, moves)
                else:
                    widgets['paths'].solved = False
                    widgets['paths'].set_text(
                        '[Hungarian] Solution Not Found!\n' + 
                        ('Deadlock Found!' if depth < 0 else f'Depth {depth}'), 
                        20,
                    )
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    moves += game.player.update(key='R')
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    moves += game.player.update(key='L')
                elif event.key in (pygame.K_w, pygame.K_UP):
                    moves += game.player.update(key='U')
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    moves += game.player.update(key='D')

        # Vẽ game
        game.floor_group.draw(window)
        game.goal_group.draw(window)
        game.object_group.draw(window)

        # Nền panel bên phải (cho rõ ràng)
        sidebar_w = SIDEBAR_COLS * int(tile)
        panel_rect = pygame.Rect(window.get_width() - sidebar_w, 0, sidebar_w, window.get_height())
        pygame.draw.rect(window, (60, 60, 70), panel_rect)

        # Vẽ UI
        pygame_widgets.update(events)
        widgets['label'].draw()
        widgets['visualizer'].draw()
        widgets['moves_label'].set_moves(f' Moves - {moves} ', 20)
        if show_solution:
            widgets['paths'].draw()

        pygame.display.update()

        if game.is_level_complete():
            print(f'Level Complete! - {moves} moves')
            widgets['level_clear'].draw()
            pygame.display.update()
            game_loop = False
            wait = True
            while wait:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        wait = False

    del game
    print('Objects cleared!\n')
    return {'keep_playing': True, 'reset': -1}


def main():
    pygame.init()
    displayIcon = pygame.image.load('img/icon.png')
    pygame.display.set_icon(displayIcon)
    pygame.display.set_caption('Sokoban')

    window = pygame.display.set_mode((BASE_W, BASE_H))
    widgets = sidebar_widgets(window)

    level = 1
    keep_playing = True

    while keep_playing:
        # Tính tile động theo level (cửa sổ cố định)
        if level >= 1:
            try:
                rows, cols = probe_level_size(f'levels/lvl{level}.dat')
            except Exception:
                rows, cols = 10, 10
            tile = compute_tile(BASE_W, BASE_H, cols, rows, SIDEBAR_COLS, MAX_TILE, MIN_TILE)
            print(f'Loading level {level} (rows={rows}, cols={cols}, tile={tile})\n')
        game_data = play_game(window, level, tile=tile, **widgets)

        keep_playing = game_data.get('keep_playing', False)
        if not keep_playing:
            pygame.quit()
            quit()

        reset = game_data.get('reset', -1)
        level = reset if reset >= 0 else min(level + 1, MAX_LEVEL)


if __name__ == '__main__':
    # wall: +, box: @, player: *, goal: X, box on goal: $, player on goal: %, empty: -
    main()
