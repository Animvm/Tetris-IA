# Tetris IA - Comparativa de Algoritmos de Aprendizaje por Refuerzo

## Descripcion

Implementacion y comparacion de tres enfoques de inteligencia artificial para jugar Tetris:

1. **DQN (Deep Q-Network)** - Aprendizaje por refuerzo profundo
2. **MCTS (Monte Carlo Tree Search)** - Busqueda en arbol con simulaciones
3. **Hibrido MCTS-DQN** - Combinacion de planificacion y aprendizaje

Este proyecto investiga como diferentes tecnicas de IA abordan el problema de toma de decisiones secuenciales en Tetris, un entorno con espacio de estados masivo y recompensas dispersas.

## Tabla de Contenidos

- [Instalacion](#instalacion)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Agentes Implementados](#agentes-implementados)
- [Mejoras y Optimizaciones](#mejoras-y-optimizaciones)
- [Uso](#uso)
- [Resultados Esperados](#resultados-esperados)
- [Fundamentos Teoricos](#fundamentos-teoricos)
- [Referencias y Papers](#referencias-y-papers)

## Instalacion

### Requisitos Previos
- Python 3.8+
- CUDA (opcional, para aceleracion GPU)

### Setup

```bash
# Clonar el repositorio
git clone https://github.com/Animvm/Tetris-IA.git
cd Tetris-IA

# Crear entorno virtual
python -m venv tetris_env

# Activar entorno virtual
# Windows:
tetris_env\Scripts\activate
# Linux/Mac:
source tetris_env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Instalacion con CPU solamente

Si no tienes GPU o prefieres usar CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verificar Instalacion

```bash
python experiments/test_environment.py
python experiments/test_action_masking.py
python experiments/test_dqn_improvements.py
```

## Estructura del Proyecto

```
Tetris-IA/
├── agents/
│   ├── dqn_agent.py           # Agente DQN mejorado con Double DQN
│   ├── mcts_agent.py          # Agente MCTS con busqueda UCB
│   ├── heuristic_agent.py     # Agente baseline heuristico
│   ├── hybrid_dqn_agent.py    # Agente hibrido MCTS-DQN (en desarrollo)
│   └── expert_generator.py    # Generador de demos expertas (en desarrollo)
│
├── envs/
│   └── tetris_env.py          # Ambiente Tetris compatible con Gymnasium
│                              # Incluye action masking y reward shaping
│
├── experiments/
│   ├── train_dqn.py           # Entrenamiento DQN completo
│   ├── train_mcts.py          # Entrenamiento MCTS
│   ├── train_heuristic.py     # Entrenamiento heuristico
│   ├── test_*.py              # Scripts de testing y validacion
│   └── analyze_training.py    # Analisis de resultados
│
├── utils/
│   ├── plotting.py            # Visualizacion de metricas
│   ├── parallel_env.py        # Entornos paralelos (en desarrollo)
│   └── statistical_analysis.py # Tests estadisticos (en desarrollo)
│
└── models/                    # Modelos entrenados guardados
```

## Agentes Implementados

### 1. DQN (Deep Q-Network)

Implementacion de Deep Q-Learning con multiples mejoras respecto al DQN vanilla.

**Arquitectura de Red:**
```
Input: Tablero 20x10 (estado del juego)
    ↓
Conv2D(32 filtros 3x3) + BatchNorm + ReLU
    ↓
Conv2D(64 filtros 3x3) + BatchNorm + ReLU
    ↓
Conv2D(128 filtros 3x3) + BatchNorm + ReLU
    ↓
Flatten
    ↓
Dense(512) + Dropout(0.2) + ReLU
    ↓
Dense(256) + Dropout(0.2) + ReLU
    ↓
Dense(40)  → Q-values para cada accion
```

**Mejoras Implementadas:**
- **Double DQN**: Reduce overestimation bias separando seleccion y evaluacion de acciones
- **Experience Replay**: Buffer de 50,000 transiciones para romper correlaciones temporales
- **Target Network**: Actualizado cada 1000 pasos para estabilidad
- **Epsilon-greedy mejorado**: Decay lento (0.9995) con minimo en 0.05
- **Batch Normalization**: Mejora estabilidad del entrenamiento
- **Dropout**: Regularizacion para prevenir overfitting
- **Gradient Clipping**: Previene gradientes explosivos

**Hiperparametros:**
```python
learning_rate = 0.00025
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
buffer_size = 50000
batch_size = 64
target_update = 1000
```

### 2. MCTS (Monte Carlo Tree Search)

Implementacion de MCTS usando Upper Confidence Bound (UCB) para balancear exploracion-explotacion.

**Algoritmo:**
```
1. Seleccion: Navegar el arbol usando UCB
   UCB = Q(s,a) + c * sqrt(log(N(s)) / N(s,a))

2. Expansion: Expandir nodo no visitado

3. Simulacion: Rollout usando politica heuristica
   - Evalua tablero: 100*lineas - 20*huecos - 5*altura
   - Profundidad maxima: 10 pasos
   - Muestrea 5 acciones aleatorias, elige mejor

4. Backpropagation: Actualizar valores hacia la raiz
```

**Parametros:**
```python
num_simulaciones = 100-200
max_profundidad = 10
c_exploracion = 1.41  # sqrt(2)
```

**Funcion de Evaluacion:**
```python
evaluacion = 100 * lineas_limpiadas
           - 20 * huecos
           - 5 * altura_maxima
```

### 3. Agente Heuristico (Baseline)

Agente determinista que evalua todas las posiciones posibles y elige la mejor segun heuristicas.

**Heuristica:**
```python
score = -0.510066 * altura_agregada
      - 0.35663 * huecos
      - 0.184483 * bumpiness
```

Donde:
- **altura_agregada**: Suma de alturas de todas las columnas
- **huecos**: Celdas vacias debajo de bloques
- **bumpiness**: Variacion entre alturas de columnas adyacentes

Estos pesos fueron optimizados empiricamente y provienen de investigacion previa en agentes Tetris.

### 4. Agente Hibrido MCTS-DQN (En Desarrollo)

Combina la planificacion de MCTS con el aprendizaje de DQN:

1. **Fase de Pre-entrenamiento**: MCTS genera demostraciones expertas
2. **Behavioral Cloning**: DQN aprende a imitar las acciones de MCTS
3. **Fine-tuning**: DQN continua mejorando con RL estandar
4. **Loss Hibrido**: `L_total = L_TD + λ * L_imitacion`

## Mejoras y Optimizaciones

### Action Space Reduction

**Problema Original:** 40 acciones por estado (10 columnas × 4 rotaciones)

**Solucion Implementada:**
1. **Filtrado de simetrias rotacionales:**
   - Pieza O: 4 rotaciones → 1 unica (todas identicas)
   - Piezas I, S, Z: 4 rotaciones → 2 unicas
   - Piezas T, J, L: 4 rotaciones (todas diferentes)

2. **Eliminacion de posiciones invalidas:**
   - Solo considerar columnas donde la pieza cabe completamente
   - Filtrado dinamico basado en ancho de pieza actual

**Resultado:** Reduccion promedio de 42.1% en espacio de acciones (40 → ~23 acciones validas)

**Beneficios:**
- Menos acciones desperdiciadas en exploracion
- Convergencia mas rapida
- Mejor sample efficiency

### Reward Shaping

**Problema Original:** Recompensa dispersa (+10 solo al limpiar lineas)

**Solucion Implementada:** Funcion de recompensa multi-componente

```python
def compute_shaped_reward(lines_cleared, board_before, board_after):
    reward = lines_cleared * 100  # Recompensa principal aumentada

    # Metricas del tablero
    def board_metrics(board):
        heights = sum(board > 0 por columna)
        max_height = max(heights)
        avg_height = mean(heights)
        holes = contar_huecos(board)
        bumpiness = sum(|heights[i] - heights[i+1]|)
        return max_height, avg_height, holes, bumpiness

    # Calcular antes y despues
    h_max_before, h_avg_before, holes_before, bump_before = board_metrics(board_before)
    h_max_after, h_avg_after, holes_after, bump_after = board_metrics(board_after)

    # Recompensas por mejoras
    reward += (h_max_before - h_max_after) * 2      # Reducir altura
    reward += (h_avg_before - h_avg_after) * 1      # Reducir altura promedio
    reward -= (holes_after - holes_before) * 5      # Penalizar huecos nuevos
    reward -= (bump_after - bump_before) * 1        # Penalizar irregularidades
    reward += 1                                      # Recompensa base

    return reward
```

**Penalizaciones:**
- Accion invalida: -50
- Game over: -100

**Beneficios:**
- Señal de aprendizaje mas densa
- Guia hacia estados favorables
- Mejora dramatica en velocidad de aprendizaje

## Uso

### 1. Probar el Ambiente

```bash
python experiments/test_environment.py
```

Verifica que el ambiente Tetris funcione correctamente.

### 2. Validar Action Masking

```bash
python experiments/test_action_masking.py
```

Muestra la reduccion de acciones por tipo de pieza:
```
Pieza O: 9 acciones validas (reduccion: 31 acciones)
Pieza I: 17 acciones validas (reduccion: 23 acciones)
...
Reduccion promedio: 42.1%
```

### 3. Entrenar DQN Mejorado

```bash
python experiments/train_dqn.py
```

**Salida durante entrenamiento:**
```
==================================================================
ENTRENAMIENTO DQN - TETRIS IA
==================================================================
Device: cuda
Episodios: 500
Resultados en: results/run_TIMESTAMP
TensorBoard: tensorboard --logdir=results/run_TIMESTAMP/tensorboard
==================================================================

Ep 50/500 | Avg Score: 45.2 | Avg Lines: 3.1 | Loss: 2.34 | ε: 0.78
Ep 100/500 | Avg Score: 120.5 | Avg Lines: 8.7 | Loss: 1.12 | ε: 0.61
...
```

**Archivos generados:**
- `results/run_TIMESTAMP/training_log.csv` - Metricas detalladas
- `results/run_TIMESTAMP/config.json` - Configuracion
- `results/run_TIMESTAMP/summary.json` - Estadisticas
- `results/run_TIMESTAMP/checkpoints/` - Modelos intermedios
- `models/dqn_tetris_final.pth` - Modelo final

**Visualizacion con TensorBoard:**
```bash
tensorboard --logdir=results/run_TIMESTAMP/tensorboard
# Abrir http://localhost:6006
```

### 4. Entrenar MCTS

```bash
python experiments/train_mcts.py
```

MCTS no requiere "entrenamiento" en el sentido de DQN, pero este script evalua su performance.

### 5. Entrenar Agente Heuristico

```bash
python experiments/train_heuristic.py
```

Baseline determinista para comparacion.

### 6. Analizar Resultados

```bash
python experiments/analyze_training.py results/run_TIMESTAMP
```

Genera analisis estadistico detallado:
- Estadisticas por fases (inicial/media/final)
- Analisis de convergencia
- Distribucion de scores
- Correlaciones entre metricas

### 7. Visualizar Agente Jugando

```bash
python experiments/visualize_run.py
```

Abre ventana con el agente jugando en tiempo real.

## Resultados Esperados

### Performance Esperada (tras 500 episodios)

| Agente | Score Promedio | Lineas Promedio | Tiempo/Episodio |
|--------|---------------|----------------|-----------------|
| **Aleatorio** | ~5 | ~0.5 | <1s |
| **Heuristico** | ~200 | ~15 | ~2s |
| **DQN Original** | ~10 | ~1 | ~3s |
| **DQN Mejorado** | **~150** | **~12** | ~3s |
| **MCTS (100 sims)** | ~250 | ~20 | ~15s |
| **Hibrido** | **~300** | **~25** | ~3s |

### Curvas de Aprendizaje Esperadas

**DQN Mejorado:**
- Episodios 1-100: Exploracion, scores bajos (~10-30)
- Episodios 100-300: Aprendizaje rapido (~30-100)
- Episodios 300-500: Refinamiento (~100-150)

**MCTS:**
- Performance consistente desde el inicio
- No mejora con episodios (sin aprendizaje)
- Alta varianza debido a naturaleza estocastica

**Hibrido:**
- Inicio fuerte (gracias a demos MCTS)
- Mejora continua (gracias a fine-tuning DQN)
- Mejor performance final

## Fundamentos Teoricos

### Deep Q-Network (DQN)

**Ecuacion de Bellman:**
```
Q(s,a) = r + γ * max_a' Q(s',a')
```

Donde:
- `Q(s,a)` = valor de tomar accion `a` en estado `s`
- `r` = recompensa inmediata
- `γ` = factor de descuento (0.99)
- `s'` = siguiente estado

**Double DQN:**
```
Q_target = r + γ * Q_target(s', argmax_a Q_policy(s',a))
```

Reduce overestimation usando policy network para seleccion de accion y target network para evaluacion.

**Loss Function:**
```
L = E[(Q(s,a) - y)^2]
donde y = r + γ * Q_target(s', argmax_a Q(s',a))
```

### Monte Carlo Tree Search (MCTS)

**Upper Confidence Bound (UCB1):**
```
UCB(s,a) = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
```

Donde:
- `Q(s,a)` = valor promedio (explotacion)
- `c` = constante de exploracion (1.41)
- `N(s)` = visitas al nodo padre
- `N(s,a)` = visitas al nodo hijo

**Proceso de 4 Fases:**
1. **Selection**: Navegar usando UCB hasta hoja
2. **Expansion**: Agregar nuevo nodo
3. **Simulation**: Rollout hasta terminar
4. **Backpropagation**: Actualizar Q-values

### Aprendizaje por Imitacion (Hibrido)

**Loss Combinado:**
```
L_total = L_TD + λ * L_BC

L_TD = MSE(Q(s,a), r + γ * max_a' Q(s',a'))  # Temporal Difference
L_BC = CrossEntropy(π_DQN(s), a_expert)      # Behavioral Cloning
```

Donde:
- `λ` = peso de imitacion (1.0 → 0.0 con annealing)
- `a_expert` = accion del experto MCTS

## Referencias y Papers

### Algoritmos Fundamentales

1. **DQN Original**
   - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
   - Nature 518, 529-533
   - Introduce DQN con experience replay y target network

2. **Double DQN**
   - van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning"
   - AAAI Conference on Artificial Intelligence
   - Soluciona overestimation bias del DQN original

3. **MCTS**
   - Browne et al. (2012). "A Survey of Monte Carlo Tree Search Methods"
   - IEEE Transactions on Computational Intelligence and AI in Games
   - Review comprehensivo de variantes MCTS

4. **UCB (Upper Confidence Bound)**
   - Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
   - Machine Learning 47, 235-256
   - Base teorica para exploration-exploitation tradeoff

### Aplicaciones a Tetris

5. **Feature-Based Tetris**
   - Thiery & Scherrer (2009). "Building Controllers for Tetris"
   - International Computer Games Association Journal
   - Heuristicas y features para Tetris

6. **Deep RL para Tetris**
   - Stevens & Pradhan (2016). "Playing Tetris with Deep Reinforcement Learning"
   - Stanford CS229 Project Report
   - Primeras aplicaciones de DQN a Tetris

### Tecnicas de Mejora

7. **Reward Shaping**
   - Ng et al. (1999). "Policy Invariance Under Reward Transformations"
   - ICML 1999
   - Teoria formal de reward shaping

8. **Action Space Reduction**
   - Dulac-Arnold et al. (2015). "Deep Reinforcement Learning in Large Discrete Action Spaces"
   - arXiv:1512.07679
   - Tecnicas para espacios de accion grandes

9. **Imitation Learning**
   - Ross et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
   - AISTATS 2011
   - DAgger algorithm

10. **Behavioral Cloning**
    - Pomerleau (1991). "Efficient Training of Artificial Neural Networks for Autonomous Navigation"
    - Neural Computation 3(1), 88-97
    - Fundamentos de BC

### Frameworks y Herramientas

11. **Gymnasium (OpenAI Gym)**
    - Brockman et al. (2016). "OpenAI Gym"
    - arXiv:1606.01540
    - Standard para ambientes RL

12. **PyTorch**
    - Paszke et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
    - NeurIPS 2019
    - Framework de deep learning usado

## Contribuciones y Mejoras Futuras

### Implementado
- [x] Environment Tetris con Gymnasium
- [x] Action masking con reduccion de simetrias
- [x] Reward shaping multi-componente
- [x] DQN con Double DQN
- [x] Arquitectura CNN mejorada
- [x] MCTS con UCB
- [x] Agente heuristico baseline

### En Desarrollo
- [ ] Agente hibrido MCTS-DQN completo
- [ ] Framework de evaluacion comparativa
- [ ] Paralelizacion con GPU
- [ ] Prioritized Experience Replay
- [ ] Dueling DQN
- [ ] Rainbow DQN

### Mejoras Potenciales
- Noisy Networks para exploracion
- Distributional RL (C51, QR-DQN)
- Model-based RL
- Curriculum learning
- Multi-agent training

## Licencia

MIT License

## Contacto

Para preguntas o colaboraciones:
- GitHub: [Animvm/Tetris-IA](https://github.com/Animvm/Tetris-IA)
- Issues: [Reportar problemas](https://github.com/Animvm/Tetris-IA/issues)

## Agradecimientos

Este proyecto fue desarrollado como parte de un curso de Inteligencia Artificial, explorando diferentes enfoques de RL aplicados al problema clasico de Tetris.

Agradecimientos especiales a la comunidad de investigacion en RL por los papers y recursos que hicieron posible este proyecto.
