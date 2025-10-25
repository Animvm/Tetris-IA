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

### Entrenar agente DQN

```bash
python experiments/train_dqn.py
```

El entrenamiento incluye:
- **Logging detallado**: Guarda todas las métricas en CSV y JSON
- **TensorBoard**: Visualización en tiempo real del entrenamiento
- **Checkpoints**: Guarda el modelo cada 100 episodios
- **Visualización periódica**: Muestra el agente jugando cada 50 episodios
- **Reproducción del mejor episodio**: Al final muestra la mejor partida

Los resultados se guardan en `results/run_TIMESTAMP/`:
- `training_log.csv`: Métricas detalladas por episodio
- `config.json`: Configuración del entrenamiento
- `summary.json`: Resumen de estadísticas
- `checkpoints/`: Modelos guardados periódicamente
- Gráficos: rewards, scores, lines, loss, epsilon, Q-values

#### Ver entrenamiento en tiempo real con TensorBoard

```bash
tensorboard --logdir=results/run_TIMESTAMP/tensorboard
```

Luego abre http://localhost:6006 en tu navegador.

### Analizar resultados de entrenamiento

```bash
# Analizar un entrenamiento específico
python experiments/analyze_training.py results/run_20250125_143022

# Analizar el entrenamiento más reciente
python experiments/analyze_training.py "results/run_*"
```

El análisis genera:
- Estadísticas por fases (inicial, media, final)
- Análisis de convergencia y tendencias
- Distribuciones de scores y líneas
- Correlaciones entre métricas
- Gráficos de aprendizaje

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