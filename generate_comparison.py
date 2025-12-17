import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# cargar datos de los tres metodos
dqn_df = pd.read_csv('dqn_parallel_metrics.csv')
mcts_df = pd.read_csv('mcts_metrics.csv')
hybrid_df = pd.read_csv('results/hybrid_20251217_011732/hybrid_metrics.csv')

print("Datos cargados:")
print(f"DQN: {len(dqn_df)} episodios")
print(f"MCTS: {len(mcts_df)} episodios")
print(f"Hybrid: {len(hybrid_df)} episodios")

# calcular promedios moviles
def moving_avg(data, window=50):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# crear figura con 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Comparacion de Agentes: DQN vs MCTS vs Hibrido', fontsize=16, fontweight='bold')

# limitar episodios para comparacion justa
max_episodes = min(len(dqn_df), len(hybrid_df))
dqn_compare = dqn_df[:max_episodes]
hybrid_compare = hybrid_df[:max_episodes]

# grafico 1: score promedio
ax1 = axes[0, 0]
window = 50
ax1.plot(moving_avg(dqn_compare['score'], window), label='DQN', linewidth=2, color='blue', alpha=0.8)
ax1.plot(moving_avg(hybrid_compare['score'], window), label='Hibrido', linewidth=2, color='green', alpha=0.8)
if len(mcts_df) > 10:
    mcts_avg = [mcts_df['score'].mean()] * max_episodes
    ax1.axhline(y=mcts_df['score'].mean(), color='red', linestyle='--', linewidth=2, label=f'MCTS (promedio)', alpha=0.8)
ax1.set_title('Score por Episodio (Media Movil 50)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episodio')
ax1.set_ylabel('Score')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# grafico 2: lineas limpiadas
ax2 = axes[0, 1]
ax2.plot(moving_avg(dqn_compare['lines'], window), label='DQN', linewidth=2, color='blue', alpha=0.8)
ax2.plot(moving_avg(hybrid_compare['lines'], window), label='Hibrido', linewidth=2, color='green', alpha=0.8)
if len(mcts_df) > 10:
    ax2.axhline(y=mcts_df['lines'].mean(), color='red', linestyle='--', linewidth=2, label=f'MCTS (promedio)', alpha=0.8)
ax2.set_title('Lineas Limpiadas (Media Movil 50)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episodio')
ax2.set_ylabel('Lineas')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# grafico 3: comparacion de ultimos 100 episodios
ax3 = axes[1, 0]
n_last = 100
dqn_last = dqn_compare['score'].tail(n_last).mean()
hybrid_last = hybrid_compare['score'].tail(n_last).mean()
mcts_avg_score = mcts_df['score'].mean()

bars = ax3.bar(['DQN', 'MCTS', 'Hibrido'],
               [dqn_last, mcts_avg_score, hybrid_last],
               color=['blue', 'red', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_title(f'Score Promedio (Ultimos {n_last} Episodios)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score Promedio')
ax3.grid(True, alpha=0.3, axis='y')

# agregar valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# grafico 4: evolucion de lambda (solo hibrido)
ax4 = axes[1, 1]
if 'imitation_weight' in hybrid_compare.columns:
    ax4.plot(hybrid_compare['episode'], hybrid_compare['imitation_weight'],
            linewidth=2, color='purple', alpha=0.8)
    ax4.set_title('Evolucion de Lambda (Peso de Imitacion)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episodio')
    ax4.set_ylabel('Lambda')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax4.text(max_episodes*0.7, 0.52, 'Transicion (λ=0.5)', fontsize=9, alpha=0.7)
else:
    # comparacion de lineas si no hay lambda
    dqn_last_lines = dqn_compare['lines'].tail(n_last).mean()
    hybrid_last_lines = hybrid_compare['lines'].tail(n_last).mean()
    mcts_avg_lines = mcts_df['lines'].mean()

    bars2 = ax4.bar(['DQN', 'MCTS', 'Hibrido'],
                   [dqn_last_lines, mcts_avg_lines, hybrid_last_lines],
                   color=['blue', 'red', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_title(f'Lineas Promedio (Ultimos {n_last} Episodios)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Lineas Promedio')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()

# guardar figura
output_dir = 'results/comparacion'
os.makedirs(output_dir, exist_ok=True)
output_path = f'{output_dir}/comparacion_completa.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGrafico guardado en: {output_path}")

# imprimir estadisticas
print("\n" + "="*60)
print("ESTADISTICAS FINALES")
print("="*60)

print(f"\nDQN (ultimos {n_last} episodios):")
print(f"  Score promedio: {dqn_last:.2f}")
print(f"  Lineas promedio: {dqn_compare['lines'].tail(n_last).mean():.2f}")
print(f"  Score maximo: {dqn_compare['score'].max():.2f}")
print(f"  Lineas maximo: {dqn_compare['lines'].max():.0f}")

print(f"\nMCTS (todos los episodios):")
print(f"  Score promedio: {mcts_avg_score:.2f}")
print(f"  Lineas promedio: {mcts_df['lines'].mean():.2f}")
print(f"  Score maximo: {mcts_df['score'].max():.2f}")
print(f"  Lineas maximo: {mcts_df['lines'].max():.0f}")

print(f"\nHibrido (ultimos {n_last} episodios):")
print(f"  Score promedio: {hybrid_last:.2f}")
print(f"  Lineas promedio: {hybrid_compare['lines'].tail(n_last).mean():.2f}")
print(f"  Score maximo: {hybrid_compare['score'].max():.2f}")
print(f"  Lineas maximo: {hybrid_compare['lines'].max():.0f}")

print("\n" + "="*60)
print("COMPARACION DE RENDIMIENTO")
print("="*60)
mejora_vs_dqn = ((hybrid_last - dqn_last) / abs(dqn_last)) * 100 if dqn_last != 0 else 0
print(f"Hibrido vs DQN: {mejora_vs_dqn:+.1f}% en score")

# no mostrar ventana, solo guardar
# plt.show()
