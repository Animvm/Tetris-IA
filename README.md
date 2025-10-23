# Tetris IA

## Descripción

Desarrollo e implementación de diferentes agentes de IA para jugar Tetris utilizando técnicas de búsqueda, reinforcement learning y MCTS. El objetivo es entrenar agentes que maximicen la puntuación y alcancen el nivel más alto posible.

## Instalación

### Requisitos previos
- Python 3.8+
- pip

### Setup
```bash
# Clonar el repositorio
git clone https://github.com/Animvm/Tetris-IA.git
cd Tetris-IA

# Crear entorno virtual
python3 -m venv tetris_env
source tetris_env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

**Nota:** Si no tienes GPU o quieres instalar PyTorch solo para CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verificar instalación
```bash
python experiments/test_environment.py
```
## Estructura del Proyecto
```
Tetris-IA/
├── agents/          # Implementación de agentes IA
├── envs/            # Ambiente de Tetris 
├── experiments/     # Scripts de entrenamiento
├── utils/           # Utilidades
└── models/          # Modelos entrenados
```

## Uso

### Probar el ambiente
```bash
python experiments/test_environment.py
```

### Entrenar agente heurístico
```bash
python experiments/train_heuristic.py
```

Esto:
- Entrena el agente durante varios episodios
- Muestra el progreso por terminal (ej: `Ep 10/200 reward 150 lines 6`)
- Genera gráficos con métricas de rendimiento en `results/heuristic_progress.png`

Los gráficos incluyen:
- Reward por episodio
- Líneas completadas
- Media móvil del desempeño

### Visualizar agente jugando
```bash
python experiments/visualize_run.py
```

Esto abre una ventana mostrando:
- El tablero de Tetris en tiempo real
- La pieza actual
- Score y líneas eliminadas
- Velocidad configurable

Los resultados se pueden guardar en `results/visual_scores.txt`.


## Referencias

- [Gymnasium Documentation](https://gymnasium.farama.org/)