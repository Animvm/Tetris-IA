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

### Verificar instalación
```bash
python experiments/test_environment.py
```

## Referencias

- [gym-tetris](https://github.com/Kautenja/gym-tetris)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

-------------------------------------------------------------------------------------------------------------------------------------
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
# o venv\Scripts\activate en Windows
```
3️⃣ Instalar dependencias (solo CPU)

    No necesitas GPU ni CUDA.
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium pygame numpy matplotlib tqdm
``
🏗️ Estructura del proyecto

IA-juegos/
│
├── envs/
│   └── tetris_env.py           # Entorno personalizado Gymnasium
│
├── agents/
│   └── heuristic_agent.py      # Agente heurístico (reglas simples)
│
├── utils/
│   └── plotting.py             # Funciones para graficar métricas
│
├── train_heuristic.py          # Entrenamiento heurístico + gráficos
├── visualize_run.py            # Visualización del agente jugando
└── README.md                   # Este archivo

🧠 Entrenar el agente heurístico

Ejecuta:

python train_heuristic.py

Esto:

    Entrena el agente durante varios episodios.

    Muestra el progreso por terminal:

Ep 10/200 reward 150 lines 6
Ep 20/200 reward 230 lines 9
...

Genera un gráfico con métricas de rendimiento:

    results/heuristic_progress.png

📊 El gráfico incluye

    Reward por episodio (total de puntos).

    Líneas completadas.

    Media móvil del desempeño.

🎮 Visualizar el juego en tiempo real

Puedes ver cómo juega el agente heurístico en el entorno Tetris:

python visualize_run.py

Esto abre una ventana con:

    El tablero de Tetris.

    La pieza actual.

    Score y líneas eliminadas.

    Velocidad configurable (delay_ms).

También puedes guardar los resultados de cada partida:

# Dentro de visualize_run.py
with open("results/visual_scores.txt", "a") as f:
    f.write(f"{ep+1},{info['score']},{info['lines']}\n")
