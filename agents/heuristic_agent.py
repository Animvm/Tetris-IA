import numpy as np
from envs.tetris_env import rotate, TETROMINOS

# calcula altura total de todas las columnas
def aggregate_height(board):
    cols = board.shape[1]
    h = 0
    for c in range(cols):
        col = board[:,c]
        nonzeros = np.where(col)[0]
        h += (board.shape[0] - nonzeros[0]) if nonzeros.size else 0
    return h

# cuenta huecos debajo de bloques
def count_holes(board):
    holes = 0
    for c in range(board.shape[1]):
        col = board[:,c]
        found_block = False
        for cell in col:
            if cell:
                found_block = True
            elif found_block and not cell:
                holes += 1
    return holes

# calcula irregularidad del tablero
def bumpiness(board):
    heights = []
    for c in range(board.shape[1]):
        col = board[:,c]
        nonzeros = np.where(col)[0]
        heights.append((board.shape[0] - nonzeros[0]) if nonzeros.size else 0)
    return sum(abs(heights[i]-heights[i+1]) for i in range(len(heights)-1))

def lines_cleared_by_board(before, after):
    return np.sum(np.all(after, axis=1)) - np.sum(np.all(before, axis=1))

# evalua calidad del tablero usando heuristica
def evaluate_board(board):
    a = aggregate_height(board)
    h = count_holes(board)
    b = bumpiness(board)
    score = -0.510066 * a - 0.35663 * h - 0.184483 * b
    return score

# agente que usa heuristica para jugar
class HeuristicAgent:
    def __init__(self, env):
        self.env = env

    # busca la mejor jugada evaluando todas las opciones
    def best_action(self, obs):
        best_score = -1e9
        best_action = 0
        # probar todas las posiciones y rotaciones
        for col in range(self.env.cols):
            for rot in range(4):
                piece = rotate(TETROMINOS[self.env.current_piece_name], rot)
                if col + piece.shape[1] > self.env.cols:
                    continue
                board_copy = obs.copy()
                y = 0
                collision = False
                # dejar caer la pieza
                while True:
                    if y + piece.shape[0] > board_copy.shape[0]:
                        break
                    slice_ = board_copy[y:y+piece.shape[0], col:col+piece.shape[1]]
                    if np.any(slice_ + piece > 1):
                        break
                    y += 1
                y -= 1
                if y < 0:
                    continue
                board_copy[y:y+piece.shape[0], col:col+piece.shape[1]] += piece
                # limpiar lineas completas
                rows_to_clear = [i for i in range(board_copy.shape[0]) if np.all(board_copy[i])]
                for r in rows_to_clear:
                    board_copy[1:r+1] = board_copy[0:r]
                    board_copy[0] = 0
                sc = evaluate_board(board_copy)
                if sc > best_score:
                    best_score = sc
                    best_action = col*4 + rot
        return best_action
