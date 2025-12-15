# envs/tetris_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import sys

# Tetrominos
TETROMINOS = {
    'I': np.array([[1,1,1,1]]),
    'O': np.array([[1,1],[1,1]]),
    'T': np.array([[0,1,0],[1,1,1]]),
    'S': np.array([[0,1,1],[1,1,0]]),
    'Z': np.array([[1,1,0],[0,1,1]]),
    'J': np.array([[1,0,0],[1,1,1]]),
    'L': np.array([[0,0,1],[1,1,1]])
}
PIECE_NAMES = list(TETROMINOS.keys())

# Rotaciones unicas por pieza para reducir espacio de acciones
# O tiene todas las rotaciones identicas
# I, S, Z tienen solo 2 rotaciones unicas (0 y 90 grados)
# T, J, L tienen las 4 rotaciones unicas
UNIQUE_ROTATIONS = {
    'O': 1,
    'I': 2,
    'S': 2,
    'Z': 2,
    'T': 4,
    'J': 4,
    'L': 4
}

def rotate(piece, k):
    # rotate k times 90 degrees
    return np.rot90(piece, k)

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, rows=20, cols=10, seed=None, render_mode=None, cell_size=24, use_action_masking=False):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.random = random.Random(seed)
        self.observation_space = spaces.Box(low=0, high=1, shape=(rows, cols), dtype=np.int8)
        self.action_space = spaces.Discrete(self.cols * 4)

        # Opcion para filtrar acciones invalidas y reducir espacio de busqueda
        self.use_action_masking = use_action_masking

        # game state
        self.current_piece_name = None
        self.current_piece = None
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0  # spawn y (top)
        self.score = 0
        self.lines_cleared_total = 0
        self.done = False

        # rendering
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.window = None
        self.clock = None
        self.surface = None
        self._colors = {
            0: (30, 30, 30),       # empty background
            1: (200, 200, 200),    # block base color (will recolor tetrominos)
        }
        # colors per piece (map name -> rgb)
        self.piece_colors = {
            'I': (0, 240, 240),
            'O': (240, 240, 0),
            'T': (160, 0, 240),
            'S': (0, 240, 0),
            'Z': (240, 0, 0),
            'J': (0, 0, 240),
            'L': (240, 160, 0)
        }

        self.reset()

    def sample_piece(self):
        name = self.random.choice(PIECE_NAMES)
        return name, TETROMINOS[name]

    def spawn_piece(self):
        self.current_piece_name, base = self.sample_piece()
        self.current_rotation = 0
        self.current_piece = base.copy()
        self.current_x = (self.cols - self.current_piece.shape[1]) // 2
        self.current_y = 0
        # immediate collision => game over
        if self.check_collision(self.current_piece, self.current_x, self.current_y):
            self.done = True

    def check_collision(self, piece, x, y):
        ph, pw = piece.shape
        if x < 0 or x + pw > self.cols:
            return True
        if y + ph > self.rows:
            return True
        board_slice = self.board[y:y+ph, x:x+pw]
        return np.any((board_slice + piece) > 1)

    def place_piece(self, piece, x):
        """
        Coloca la pieza en el tablero y limpia las lineas completas.
        Retorna (valido, lineas_limpiadas) en lugar de reward.
        El reward se calcula en step() con reward shaping.
        """
        # Dejar caer la pieza hasta que colisione
        y = 0
        while not self.check_collision(piece, x, y):
            y += 1
        y -= 1
        if y < 0:
            return False, 0
        ph, pw = piece.shape
        self.board[y:y+ph, x:x+pw] += piece

        # Limpiar lineas completas
        lines = 0
        full_rows = [i for i in range(self.rows) if np.all(self.board[i])]
        for r in full_rows:
            self.board[1:r+1] = self.board[0:r]
            self.board[0] = 0
            lines += 1

        self.lines_cleared_total += lines
        return True, lines

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"score": self.score, "lines": self.lines_cleared_total}

        # Guardar estado del tablero antes de la accion
        board_before = self.board.copy()

        col = int(action // 4)
        rot = int(action % 4)

        # Validar que la columna este en rango
        if col >= self.cols:
            return self._get_obs(), -50, False, False, {"score": self.score, "lines": self.lines_cleared_total}

        piece = rotate(TETROMINOS[self.current_piece_name], rot)

        # Validar que la pieza quepa horizontalmente
        if col + piece.shape[1] > self.cols:
            return self._get_obs(), -50, False, False, {"score": self.score, "lines": self.lines_cleared_total}

        valid, lines = self.place_piece(piece, col)

        if not valid:
            # Game over: pieza no se puede colocar
            self.done = True
            reward = -100
        else:
            # Calcular recompensa con reward shaping
            reward = self.compute_shaped_reward(lines, board_before, self.board)
            self.score += reward
            self.spawn_piece()

        obs = self._get_obs()
        info = {"score": self.score, "lines": self.lines_cleared_total}
        return obs, reward, self.done, False, info

    def _get_obs(self):
        return self.board.copy()

    def compute_shaped_reward(self, lines_cleared, board_before, board_after):
        """
        Calcula recompensa con multiples componentes para mejor aprendizaje.
        Combina recompensas por lineas limpiadas con heuristicas de calidad del tablero.

        Componentes:
        - Lineas limpiadas: +100 por linea (principal)
        - Reduccion de altura: +2 por unidad reducida
        - Penalizacion por huecos: -5 por cada hueco nuevo
        - Penalizacion por bumpiness: -1 por incremento en variacion de alturas
        - Recompensa base: +1 por colocacion valida
        """
        reward = lines_cleared * 100

        def board_metrics(board):
            heights = np.sum(board > 0, axis=0)
            max_height = np.max(heights) if heights.size > 0 else 0
            avg_height = np.mean(heights) if heights.size > 0 else 0

            # Contar huecos: celdas vacias debajo de bloques
            holes = 0
            for col in range(board.shape[1]):
                found_block = False
                for row in range(board.shape[0]):
                    if board[row, col] > 0:
                        found_block = True
                    elif found_block:
                        holes += 1

            # Bumpiness: diferencia de altura entre columnas adyacentes
            bumpiness = sum(abs(heights[i] - heights[i+1])
                           for i in range(len(heights)-1))

            return max_height, avg_height, holes, bumpiness

        h_max_before, h_avg_before, holes_before, bump_before = board_metrics(board_before)
        h_max_after, h_avg_after, holes_after, bump_after = board_metrics(board_after)

        # Recompensas por mejoras en el tablero
        reward += (h_max_before - h_max_after) * 2
        reward += (h_avg_before - h_avg_after) * 1
        reward -= (holes_after - holes_before) * 5
        reward -= (bump_after - bump_before) * 1
        reward += 1

        return reward

    def reset(self, seed=None, options=None):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.score = 0
        self.lines_cleared_total = 0
        self.done = False
        if seed is not None:
            self.random = random.Random(seed)
        self.spawn_piece()
        return self._get_obs(), {}

    def get_valid_actions(self):
        """
        Retorna lista de indices de acciones validas para la pieza actual.
        Filtra por rotaciones simetricas y posiciones invalidas.
        Reduce el espacio de acciones de 40 a aproximadamente 24-28.
        """
        valid_actions = []
        piece_name = self.current_piece_name
        max_rotations = UNIQUE_ROTATIONS[piece_name]

        for rot in range(max_rotations):
            piece = rotate(TETROMINOS[piece_name], rot)
            # Solo considerar columnas donde la pieza cabe completamente
            max_col = self.cols - piece.shape[1]
            for col in range(max_col + 1):
                action = col * 4 + rot
                valid_actions.append(action)

        return valid_actions

    def get_action_mask(self):
        """
        Retorna mascara booleana de acciones validas (shape: 40,).
        Util para algoritmos que requieren formato de mascara.
        """
        mask = np.zeros(40, dtype=bool)
        valid = self.get_valid_actions()
        mask[valid] = True
        return mask

    # ---------- PYGAME RENDERING ----------
    def _init_pygame(self):
        if self.window is not None:
            return
        pygame.init()
        w = self.cols * self.cell_size + 200  # extra for side info
        h = self.rows * self.cell_size
        self.window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Tetris Gym")
        self.surface = pygame.Surface((w, h))
        self.clock = pygame.time.Clock()
        # font
        try:
            self._font = pygame.font.SysFont("Arial", 16)
        except Exception:
            self._font = pygame.font.Font(None, 16)

    def render(self):
        # compatible simple API: call render() each step
        if self.render_mode not in ("human", "rgb_array"):
            return
        if self.window is None:
            self._init_pygame()

        # handle events so window remains responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # allow graceful close
                self.close()
                return

        # background
        self.surface.fill((10, 10, 10))

        # draw board cells
        board_surface = pygame.Surface((self.cols * self.cell_size, self.rows * self.cell_size))
        board_surface.fill((20, 20, 20))
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r, c]
                rect = pygame.Rect(c*self.cell_size, r*self.cell_size, self.cell_size-1, self.cell_size-1)
                if cell:
                    # We don't store piece id per cell; use generic color
                    pygame.draw.rect(board_surface, (200,200,200), rect)
                else:
                    pygame.draw.rect(board_surface, (30,30,30), rect, 1)

        # draw current falling piece at spawn-y (visual only)
        if self.current_piece is not None:
            ph, pw = self.current_piece.shape
            color = self.piece_colors.get(self.current_piece_name, (200,200,200))
            for ry in range(ph):
                for rx in range(pw):
                    if self.current_piece[ry, rx]:
                        draw_x = (self.current_x + rx) * self.cell_size
                        draw_y = (self.current_y + ry) * self.cell_size
                        rect = pygame.Rect(draw_x, draw_y, self.cell_size-1, self.cell_size-1)
                        pygame.draw.rect(board_surface, color, rect)

        self.surface.blit(board_surface, (0,0))

        # side panel (score / lines)
        info_x = self.cols * self.cell_size + 10
        score_surf = self._font.render(f"Score: {self.score}", True, (230,230,230))
        lines_surf = self._font.render(f"Lines: {self.lines_cleared_total}", True, (230,230,230))
        self.surface.blit(score_surf, (info_x, 10))
        self.surface.blit(lines_surf, (info_x, 40))

        # instructions
        inst = self._font.render("Close window to stop", True, (180,180,180))
        self.surface.blit(inst, (info_x, self.rows*self.cell_size - 30))

        # blit to window and update
        self.window.blit(self.surface, (0,0))
        pygame.display.flip()
        # cap fps
        self.clock.tick(self.metadata.get("render_fps", 10))

    def get_rgb_array(self):
        # return numpy array of current window (useful if render_mode=='rgb_array')
        if self.window is None:
            self._init_pygame()
        return pygame.surfarray.array3d(self.window).transpose([1,0,2])

    def close(self):
        try:
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
        except Exception:
            pass
        self.window = None
        self.surface = None
        self.clock = None
