import matplotlib.pyplot as plt
import numpy as np
import os

# calcula promedio movil de datos
def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# genera graficos comparativos entre agentes
def plot_comparison(results_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Comparacion de Agentes Tetris IA', fontsize=16, fontweight='bold')

    for name, metrics in results_dict.items():
        episodes = range(1, len(metrics['scores']) + 1)
        window = min(50, len(metrics['scores']) // 10)
        if len(metrics['scores']) > window:
            ma = moving_average(metrics['scores'], window)
            axes[0, 0].plot(episodes[window-1:], ma, label=name.upper(), linewidth=2)
        else:
            axes[0, 0].plot(episodes, metrics['scores'], label=name.upper(), linewidth=2)

    axes[0, 0].set_title('Score (Media Movil)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for name, metrics in results_dict.items():
        episodes = range(1, len(metrics['lines']) + 1)
        window = min(50, len(metrics['lines']) // 10)
        if len(metrics['lines']) > window:
            ma = moving_average(metrics['lines'], window)
            axes[0, 1].plot(episodes[window-1:], ma, label=name.upper(), linewidth=2)
        else:
            axes[0, 1].plot(episodes, metrics['lines'], label=name.UPPER(), linewidth=2)

    axes[0, 1].set_title('Lineas Limpiadas (Media Movil)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Lineas')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for name, metrics in results_dict.items():
        cumsum = np.cumsum(metrics['scores'])
        axes[1, 0].plot(cumsum, label=name.upper(), linewidth=2)

    axes[1, 0].set_title('Sample Efficiency (Score Acumulado)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Score Acumulado')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    agents = list(results_dict.keys())
    comp_times = [np.mean(results_dict[a]['computation_time']) for a in agents]
    comp_stds = [np.std(results_dict[a]['computation_time']) for a in agents]

    axes[1, 1].bar(range(len(agents)), comp_times, yerr=comp_stds,
                   tick_label=[a.upper() for a in agents],
                   capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_title('Tiempo Promedio por Episodio', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Tiempo (segundos)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    final_scores = []
    final_stds = []
    for name in agents:
        n = min(100, len(results_dict[name]['scores']))
        final_scores.append(np.mean(results_dict[name]['scores'][-n:]))
        final_stds.append(np.std(results_dict[name]['scores'][-n:]))

    axes[2, 0].bar(range(len(agents)), final_scores, yerr=final_stds,
                   tick_label=[a.upper() for a in agents],
                   capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2, 0].set_title('Performance Final (Ultimos 100 Episodios)', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Score Promedio')
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    final_pieces = []
    for name in agents:
        n = min(100, len(results_dict[name]['pieces']))
        final_pieces.append(np.mean(results_dict[name]['pieces'][-n:]))

    axes[2, 1].bar(range(len(agents)), final_pieces,
                   tick_label=[a.upper() for a in agents],
                   alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2, 1].set_title('Piezas Promedio Colocadas', fontsize=12, fontweight='bold')
    axes[2, 1].set_ylabel('Piezas')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot de comparacion guardado: {save_path}")

def plot_learning_curves_detailed(results_dict, save_dir):
    for name, metrics in results_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Curvas de Aprendizaje: {name.upper()}', fontsize=14, fontweight='bold')

        episodes = range(1, len(metrics['scores']) + 1)

        axes[0, 0].plot(episodes, metrics['scores'], alpha=0.3, color='blue')
        window = min(50, len(metrics['scores']) // 10)
        if len(metrics['scores']) > window:
            ma = moving_average(metrics['scores'], window)
            axes[0, 0].plot(episodes[window-1:], ma, color='darkblue', linewidth=2)
        axes[0, 0].set_title('Score por Episodio')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(episodes, metrics['lines'], alpha=0.3, color='green')
        if len(metrics['lines']) > window:
            ma = moving_average(metrics['lines'], window)
            axes[0, 1].plot(episodes[window-1:], ma, color='darkgreen', linewidth=2)
        axes[0, 1].set_title('Lineas Limpiadas por Episodio')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Lineas')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(metrics['scores'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(np.mean(metrics['scores']), color='red', linestyle='--',
                          linewidth=2, label=f'Media: {np.mean(metrics["scores"]):.1f}')
        axes[1, 0].set_title('Distribucion de Scores')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(episodes, metrics['pieces'], alpha=0.3, color='orange')
        if len(metrics['pieces']) > window:
            ma = moving_average(metrics['pieces'], window)
            axes[1, 1].plot(episodes[window-1:], ma, color='darkorange', linewidth=2)
        axes[1, 1].set_title('Piezas Colocadas por Episodio')
        axes[1, 1].set_xlabel('Episodio')
        axes[1, 1].set_ylabel('Piezas')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{name}_learning_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Curvas de aprendizaje guardadas: {save_path}")
