import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.tetris_env import TetrisEnv, UNIQUE_ROTATIONS

def test_action_masking():
    print("Probando action masking...")
    print("="*60)

    env = TetrisEnv(use_action_masking=True)

    piece_types = ['O', 'I', 'S', 'Z', 'T', 'J', 'L']

    for piece_name in piece_types:
        env.current_piece_name = piece_name
        valid_actions = env.get_valid_actions()

        max_rotations = UNIQUE_ROTATIONS[piece_name]

        print(f"\nPieza {piece_name}:")
        print(f"  Rotaciones unicas: {max_rotations}")
        print(f"  Acciones validas: {len(valid_actions)}")
        print(f"  Reduccion: {40 - len(valid_actions)} acciones eliminadas")

        assert len(valid_actions) == len(set(valid_actions)), "Acciones duplicadas encontradas"

        for action in valid_actions:
            assert 0 <= action < 40, f"Accion fuera de rango: {action}"

    print("\n" + "="*60)
    print("Todas las pruebas pasaron correctamente")

    env.reset()
    total_actions_before = 40
    total_actions_after = 0
    num_pieces = len(piece_types)

    for piece_name in piece_types:
        env.current_piece_name = piece_name
        valid_actions = env.get_valid_actions()
        total_actions_after += len(valid_actions)

    avg_actions = total_actions_after / num_pieces
    reduction_pct = (1 - avg_actions / total_actions_before) * 100

    print(f"\nEstadisticas:")
    print(f"  Acciones por pieza (original): {total_actions_before}")
    print(f"  Acciones por pieza (promedio con masking): {avg_actions:.1f}")
    print(f"  Reduccion promedio: {reduction_pct:.1f}%")

if __name__ == "__main__":
    test_action_masking()
