# Guía de Uso - Tetris IA

Este documento explica cómo usar todos los componentes implementados en el proyecto.

---

## Estructura del Proyecto

```
Tetris-IA/
├── envs/
│   └── tetris_env.py          # Entorno Tetris con action masking y reward shaping
├── agents/
│   ├── dqn_agent.py           # DQN mejorado con Double DQN
│   ├── mcts_agent.py          # MCTS con UCB
│   ├── expert_generator.py   # Generador de datos expertos con MCTS
│   └── hybrid_dqn_agent.py    # DQN híbrido que aprende de MCTS
├── utils/
│   ├── parallel_env.py        # Entornos paralelos para aceleración
│   ├── statistical_analysis.py # Análisis estadístico comparativo
│   └── comparison_plots.py     # Visualizaciones comparativas
└── experiments/
    ├── train_dqn.py            # Entrenar DQN estándar
    ├── train_dqn_parallel.py   # Entrenar DQN con paralelización
    ├── train_mcts.py           # Evaluar MCTS puro
    ├── train_hybrid.py         # Entrenar agente híbrido
    ├── comparative_evaluation.py # Comparación completa de los 3 agentes
    ├── test_action_masking.py  # Verificar action masking
    ├── test_dqn_improvements.py # Verificar mejoras DQN
    └── test_parallel.py        # Verificar entrenamiento paralelo
```

---

## Componentes Implementados

### FASE 1: Action Space Reduction (42.1% de reducción)

**Archivo**: `envs/tetris_env.py`

**Características**:
- Filtrado de rotaciones simétricas (O: 4→1, I/S/Z: 4→2)
- Validación dinámica de columnas según ancho de pieza
- Métodos `get_valid_actions()` y `get_action_mask()`

**Uso**:
```python
from envs.tetris_env import TetrisEnv

env = TetrisEnv(use_action_masking=True)
obs, _ = env.reset()

# Obtener acciones válidas
valid_actions = env.get_valid_actions()  # Lista de índices válidos
print(f"Acciones válidas: {len(valid_actions)}/40")
```

**Prueba**:
```bash
python experiments/test_action_masking.py
```

---

### FASE 2: DQN Improvements

**Archivo**: `agents/dqn_agent.py`

**Mejoras implementadas**:
1. **Reward Shaping**: Recompensas multicomponente (líneas, altura, huecos, bumpiness)
2. **Arquitectura mejorada**: 3 capas conv + BatchNorm + Dropout
3. **Double DQN**: Reduce sobreestimación de Q-values
4. **Hiperparámetros optimizados**:
   - Learning rate: 0.00025
   - Buffer: 50,000 transiciones
   - Batch size: 64
   - Target update: cada 1,000 pasos
   - Epsilon decay: 0.9995

**Uso**:
```python
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

env = TetrisEnv(use_action_masking=True)
agent = DQNAgent(
    env,
    lr=0.00025,
    buffer_size=50000,
    batch_size=64,
    target_update=1000,
    use_double_dqn=True
)

# Entrenar
obs, _ = env.reset()
valid_actions = env.get_valid_actions()
action = agent.select_action(obs, training=True, valid_actions=valid_actions)
```

**Entrenamiento estándar**:
```bash
python experiments/train_dqn.py
```

**Prueba de mejoras**:
```bash
python experiments/test_dqn_improvements.py
```

---

### FASE 3: Hybrid MCTS-DQN

**Archivos**: `agents/expert_generator.py`, `agents/hybrid_dqn_agent.py`

**Enfoque**:
1. MCTS genera demostraciones expertas (200 simulaciones)
2. DQN aprende por imitación + RL combinados
3. Pre-entrenamiento con imitación pura
4. Fine-tuning con loss híbrido

**Uso completo**:
```bash
# Entrenar agente híbrido (genera datos MCTS automáticamente)
python experiments/train_hybrid.py
```

**Uso programático**:
```python
from envs.tetris_env import TetrisEnv
from agents.hybrid_dqn_agent import HybridDQNAgent
from agents.expert_generator import MCTSExpertGenerator

# 1. Generar datos expertos
env = TetrisEnv(use_action_masking=True)
generator = MCTSExpertGenerator(env, num_simulations=200)
dataset, expert_path = generator.generate_dataset(
    num_episodes=100,
    min_score=50
)

# 2. Crear agente híbrido
agent = HybridDQNAgent(
    env,
    expert_data_path=expert_path,
    imitation_weight=1.0,
    lr=0.0001,
    use_double_dqn=True
)

# 3. Pre-entrenar con imitación
for _ in range(500):
    for _ in range(10):
        agent.train_step_hybrid()

# 4. Fine-tuning con RL
# (loop de entrenamiento normal usando train_step_hybrid())
```

**Parámetros configurables**:
- `num_simulations`: simulaciones MCTS (default: 200)
- `min_score`: filtro de calidad para episodios expertos (default: 50)
- `imitation_weight`: peso inicial del loss de imitación (default: 1.0)
- Se aplica annealing automático (0.9999 por paso)

---

### FASE 4: Comparative Evaluation

**Archivos**: `experiments/comparative_evaluation.py`, `utils/statistical_analysis.py`, `utils/comparison_plots.py`

**Características**:
- Entrena/evalúa los 3 agentes con mismas condiciones
- Análisis estadístico (Mann-Whitney U, Cohen's d)
- Visualizaciones comparativas (6 subplots)
- Reportes automatizados

**Uso**:
```bash
# Ejecutar comparación completa (toma ~6-8 horas)
python experiments/comparative_evaluation.py
```

**Resultados generados**:
```
results/comparison_YYYYMMDD_HHMMSS/
├── dqn_final.pth
├── hybrid_final.pth
├── dqn_metrics.csv
├── mcts_metrics.csv
├── hybrid_metrics.csv
├── statistical_report.txt
├── comparison_overview.png
├── dqn_learning_curves.png
├── mcts_learning_curves.png
└── hybrid_learning_curves.png
```

**Métricas evaluadas**:
- Score promedio (últimos 100 episodios)
- Líneas limpiadas promedio
- Piezas colocadas promedio
- Sample efficiency (AUC)
- Tiempo computacional por episodio
- Significancia estadística entre agentes

---

### FASE 5: GPU Parallelization

**Archivos**: `utils/parallel_env.py`, `experiments/train_dqn_parallel.py`

**Características**:
- Ejecución paralela de 8 entornos (configurable)
- Batch inference en GPU para selección de acciones
- Speedup esperado: 5-10x vs entrenamiento secuencial

**Uso**:
```bash
# Entrenar con paralelización (recomendado para RTX 3060)
python experiments/train_dqn_parallel.py
```

**Prueba rápida**:
```bash
python experiments/test_parallel.py
```

**Configuración**:
```python
# Ajustar según CPU disponibles
agent, metrics = train_dqn_parallel(
    episodes=2000,
    num_parallel_envs=8,  # Ajustar según CPU cores
    save_interval=500
)
```

**Nota**: El speedup real depende de:
- Número de cores CPU (para workers)
- GPU para batch inference
- Overhead de comunicación entre procesos

---

## Flujos de Trabajo Recomendados

### 1. Verificación Rápida (5-10 minutos)
```bash
python experiments/test_action_masking.py
python experiments/test_dqn_improvements.py
python experiments/test_parallel.py
```

### 2. Entrenar DQN Mejorado (2-3 horas)
```bash
# Opción A: Secuencial
python experiments/train_dqn.py

# Opción B: Paralelo (recomendado)
python experiments/train_dqn_parallel.py
```

### 3. Entrenar Agente Híbrido (4-5 horas)
```bash
python experiments/train_hybrid.py
```

### 4. Comparación Completa (6-8 horas)
```bash
python experiments/comparative_evaluation.py
```

---

## Resultados Esperados

### DQN Mejorado (post-mejoras)
- Score promedio (últimos 100 eps): **50-100**
- Líneas promedio: **5-10**
- Tiempo por episodio: **~2-3 segundos**

### MCTS Puro
- Score promedio: **150-300**
- Líneas promedio: **15-30**
- Tiempo por episodio: **~10-20 segundos** (100 sims), **~40-60 segundos** (200 sims)

### Híbrido MCTS→DQN
- Score promedio: **100-200**
- Líneas promedio: **10-20**
- Tiempo por episodio: **~2-3 segundos**
- Converge más rápido que DQN puro

---

## Solución de Problemas

### Error: CUDA out of memory
```python
# Reducir batch_size en el agente
agent = DQNAgent(env, batch_size=32)  # En lugar de 64
```

### Entrenamiento muy lento
```bash
# Usar versión paralela
python experiments/train_dqn_parallel.py

# O reducir num_simulations en MCTS
generator = MCTSExpertGenerator(env, num_simulations=100)
```

### Scores negativos al inicio
- Es normal debido a reward shaping
- Los agentes mejoran después de ~200-500 episodios
- Las penalizaciones por huecos/altura se compensan con líneas

### No se detecta GPU
```python
import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar RTX 3060
```

---

## Configuración Avanzada

### Ajustar reward shaping
Editar `envs/tetris_env.py`, método `compute_shaped_reward()`:
```python
reward = lines_cleared * 100  # Aumentar/reducir peso de líneas
reward += (h_max_before - h_max_after) * 2  # Peso de altura
reward -= (holes_after - holes_before) * 5   # Penalización huecos
reward -= (bump_after - bump_before) * 1     # Penalización bumpiness
```

### Ajustar hiperparámetros DQN
```python
agent = DQNAgent(
    env,
    lr=0.0001,              # Learning rate más bajo para estabilidad
    epsilon_decay=0.999,    # Decay más rápido para menos exploración
    buffer_size=100000,     # Buffer más grande (requiere más RAM)
    target_update=500       # Actualización más frecuente (menos estable)
)
```

### Ajustar MCTS
```python
mcts = MCTSAgent(
    env,
    num_simulations=500,    # Más simulaciones = mejor pero más lento
    max_profundidad=15,     # Mayor profundidad = más lookahead
    c_puct=1.4              # Más exploración (default: sqrt(2) ≈ 1.414)
)
```

---

## Análisis de Resultados

Después de ejecutar `comparative_evaluation.py`, revisar:

1. **Statistical Report** (`statistical_report.txt`):
   - Métricas resumen por agente
   - Tests de significancia estadística
   - Intervalos de confianza

2. **Comparison Overview** (`comparison_overview.png`):
   - Curvas de aprendizaje
   - Sample efficiency
   - Tiempo computacional
   - Performance final

3. **Learning Curves** (individuales):
   - Progreso episodio a episodio
   - Distribución de scores
   - Piezas colocadas

---

## Referencias

Consultar `README.md` para:
- Detalles técnicos de cada agente
- 12 papers académicos citados
- Fundamentos matemáticos
- Arquitectura del proyecto
