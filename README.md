# IA para Juegos - Tetris

## Descripción

Este proyecto compara tres agentes de inteligencia artificial para jugar Tetris: DQN (aprendizaje profundo), MCTS (búsqueda en árbol) y un agente híbrido que combina ambos enfoques.

## Instalación

1. Crear el entorno virtual:
```bash
python -m venv tetris_env
```

2. Activar el entorno:
```bash
# Windows:
tetris_env\Scripts\activate
# Linux/Mac:
source tetris_env/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Entrenar los agentes:
```bash
python experiments/train_dqn.py
python experiments/train_mcts.py
python experiments/train_hybrid.py
```

Ver el agente jugando:
```bash
python experiments/visualize_run.py
```

Analizar resultados:
```bash
python experiments/analyze_training.py results/run_TIMESTAMP
```
