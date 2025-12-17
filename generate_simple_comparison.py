import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# cargar datos
dqn_df = pd.read_csv('dqn_parallel_metrics.csv')
mcts_df = pd.read_csv('mcts_metrics.csv')
hybrid_df = pd.read_csv('results/hybrid_20251217_011732/hybrid_metrics.csv')

# calcular media movil
def moving_avg(data, window=50):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# limitar a 1500 episodios para comparacion
max_ep = 1500
dqn_compare = dqn_df[:max_ep]
hybrid_compare = hybrid_df[:max_ep]

# figura simple con 2 graficos principales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Comparacion de Agentes en Tetris', fontsize=14, fontweight='bold')

# grafico 1: curva de aprendizaje (score)
ax1 = axes[0]
window = 50
ax1.plot(moving_avg(dqn_compare['score'], window), label='DQN', linewidth=2.5, color='#3498db')
ax1.plot(moving_avg(hybrid_compare['score'], window), label='Hibrido MCTS-DQN', linewidth=2.5, color='#2ecc71')
ax1.axhline(y=mcts_df['score'].mean(), color='#e74c3c', linestyle='--', linewidth=2.5, label='MCTS (promedio)')
ax1.set_title('Curva de Aprendizaje', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episodio', fontsize=11)
ax1.set_ylabel('Score (Media Movil 50)', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_xlim(0, max_ep)

# grafico 2: comparacion final (barras)
ax2 = axes[1]
n_last = 100
dqn_final = dqn_compare['score'].tail(n_last).mean()
mcts_final = mcts_df['score'].mean()
hybrid_final = hybrid_compare['score'].tail(n_last).mean()

x_pos = np.arange(3)
values = [dqn_final, mcts_final, hybrid_final]
colors = ['#3498db', '#e74c3c', '#2ecc71']
labels = ['DQN', 'MCTS', 'Hibrido']

bars = ax2.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_title(f'Rendimiento Final (ultimos {n_last} eps)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score Promedio', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=11)
ax2.grid(True, alpha=0.2, axis='y', linestyle='--')

# agregar valores sobre barras
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 20,
            f'{val:.0f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()

# guardar
output_path = 'results/comparacion/comparacion_simple.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Grafico simple guardado en: {output_path}")

# crear grafico solo de lambda
fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
if 'imitation_weight' in hybrid_df.columns:
    ax.plot(hybrid_df['episode'], hybrid_df['imitation_weight'],
           linewidth=2.5, color='#9b59b6')
    ax.fill_between(hybrid_df['episode'], 0, hybrid_df['imitation_weight'],
                    alpha=0.3, color='#9b59b6')
    ax.set_title('Evolucion del Peso de Imitacion (Lambda) en Modelo Hibrido',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Episodio', fontsize=11)
    ax.set_ylabel('Lambda (Peso de Imitacion)', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(1000, 0.52, 'Punto medio (λ=0.5)', fontsize=10, color='red', fontweight='bold')

    # agregar anotaciones
    ax.annotate('100% Imitacion', xy=(0, 1.0), xytext=(200, 0.85),
               fontsize=10, ha='left',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('Mayor Autonomia', xy=(1500, hybrid_df['imitation_weight'].iloc[-1]),
               xytext=(1200, 0.4), fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.tight_layout()
    lambda_path = 'results/comparacion/lambda_evolution.png'
    plt.savefig(lambda_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Grafico lambda guardado en: {lambda_path}")

print("\nGraficos generados exitosamente!")
