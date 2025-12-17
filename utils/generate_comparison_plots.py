import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# carga metricas de diferentes entrenamientos
def load_metrics():
    dqn_pattern = 'results/run_*/dqn_parallel_metrics.csv'
    dqn_files = glob(dqn_pattern)
    if not dqn_files:
        dqn_pattern = 'results/dqn_parallel_*/dqn_parallel_metrics.csv'
        dqn_files = glob(dqn_pattern)
    dqn = pd.read_csv(dqn_files[0]) if dqn_files else None

    mcts = pd.read_csv('results/mcts_metrics.csv') if os.path.exists('results/mcts_metrics.csv') else None

    hybrid_pattern = 'results/hybrid_*/hybrid_metrics.csv'
    hybrid_files = glob(hybrid_pattern)
    hybrid = pd.read_csv(hybrid_files[0]) if hybrid_files else None

    return {'DQN': dqn, 'MCTS': mcts, 'Hybrid': hybrid}

# grafica comparacion de scores
def plot_scores(data):
    plt.figure(figsize=(12, 6))
    for name, df in data.items():
        if df is not None:
            ma = df['score'].rolling(10, min_periods=1).mean()
            plt.plot(df['episode'], ma, label=name, linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Score (Media Movil 10)')
    plt.title('Comparacion de Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/comparison_plots/scores.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_lines(data):
    plt.figure(figsize=(12, 6))
    for name, df in data.items():
        if df is not None:
            ma = df['lines'].rolling(10, min_periods=1).mean()
            plt.plot(df['episode'], ma, label=name, linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Lineas (Media Movil 10)')
    plt.title('Lineas Limpiadas Promedio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/comparison_plots/lines.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_boxplots(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scores = [df['score'] for df in data.values() if df is not None]
    labels = [name for name, df in data.items() if df is not None]
    ax1.boxplot(scores, labels=labels)
    ax1.set_ylabel('Score')
    ax1.set_title('Distribucion de Scores')
    ax1.grid(True, alpha=0.3)

    lines = [df['lines'] for df in data.values() if df is not None]
    ax2.boxplot(lines, labels=labels)
    ax2.set_ylabel('Lineas')
    ax2.set_title('Distribucion de Lineas')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/comparison_plots/boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(data):
    with open('results/comparison_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPARACION DE AGENTES - TETRIS IA\n")
        f.write("="*70 + "\n\n")

        for name, df in data.items():
            if df is not None:
                f.write(f"\n{name}:\n")
                f.write(f"  Score promedio: {df['score'].mean():.2f} ± {df['score'].std():.2f}\n")
                f.write(f"  Score maximo: {df['score'].max():.2f}\n")
                f.write(f"  Score minimo: {df['score'].min():.2f}\n")
                f.write(f"  Lineas promedio: {df['lines'].mean():.2f} ± {df['lines'].std():.2f}\n")
                f.write(f"  Lineas maximo: {df['lines'].max():.0f}\n")
                f.write(f"  Episodios: {len(df)}\n")

        f.write("\n" + "="*70 + "\n")

def main():
    print("Generando graficos comparativos...")
    os.makedirs('results/comparison_plots', exist_ok=True)

    data = load_metrics()

    plot_scores(data)
    print("Grafico de scores")

    plot_lines(data)
    print("Grafico de lineas")

    plot_boxplots(data)
    print("Box plots")

    generate_report(data)
    print("Reporte estadistico")

    print(f"\nGraficos en: results/comparison_plots/")
    print(f"Reporte en: results/comparison_report.txt")

if __name__ == '__main__':
    main()
