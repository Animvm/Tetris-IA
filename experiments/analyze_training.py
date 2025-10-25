"""
Script para analizar resultados de entrenamientos DQN
Uso: python experiments/analyze_training.py results/run_TIMESTAMP
"""
import sys
sys.path.append('.')

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training(run_dir):
    """
    Analiza los resultados de un entrenamiento y genera reportes

    Args:
        run_dir: directorio con los resultados del entrenamiento
    """
    run_path = Path(run_dir)

    if not run_path.exists():
        print(f"Error: No se encontró el directorio {run_dir}")
        return

    # Cargar configuración
    config_path = run_path / "config.json"
    if not config_path.exists():
        print(f"Error: No se encontró config.json en {run_dir}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Cargar datos de entrenamiento
    log_path = run_path / "training_log.csv"
    if not log_path.exists():
        print(f"Error: No se encontró training_log.csv en {run_dir}")
        return

    df = pd.read_csv(log_path)

    # Cargar resumen si existe
    summary_path = run_path / "summary.json"
    summary = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)

    # Imprimir información general
    print("="*80)
    print(f"ANÁLISIS DE ENTRENAMIENTO: {run_path.name}")
    print("="*80)
    print(f"\nConfiguración:")
    print(f"  Timestamp: {config['timestamp']}")
    print(f"  Episodios: {config['episodes']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Gamma: {config['gamma']}")
    print(f"  Epsilon decay: {config['epsilon_decay']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Device: {config['device']}")

    if summary:
        print(f"\nResumen del entrenamiento:")
        print(f"  Tiempo total: {int(summary['training_time']//60)}m {int(summary['training_time']%60)}s")
        print(f"  Mejor episodio: {summary['best_episode']}")
        print(f"    - Score: {summary['best_score']}")
        print(f"    - Líneas: {summary['best_lines']}")
        print(f"  Promedios (últimos 50):")
        print(f"    - Score: {summary['avg_score_last_50']:.1f}")
        print(f"    - Líneas: {summary['avg_lines_last_50']:.1f}")

    # Estadísticas por fases del entrenamiento
    print(f"\n{'='*80}")
    print("ANÁLISIS POR FASES")
    print("="*80)

    n_episodes = len(df)
    phases = {
        'Inicial (1-100)': (0, min(100, n_episodes)),
        'Media (101-250)': (100, min(250, n_episodes)),
        'Final (251+)': (250, n_episodes)
    }

    for phase_name, (start, end) in phases.items():
        if start >= n_episodes:
            continue
        phase_df = df.iloc[start:end]
        if len(phase_df) == 0:
            continue

        print(f"\n{phase_name}:")
        print(f"  Episodios: {len(phase_df)}")
        print(f"  Score promedio: {phase_df['score'].mean():.1f} ± {phase_df['score'].std():.1f}")
        print(f"  Líneas promedio: {phase_df['lines'].mean():.1f} ± {phase_df['lines'].std():.1f}")
        print(f"  Reward promedio: {phase_df['reward'].mean():.1f} ± {phase_df['reward'].std():.1f}")
        print(f"  Loss promedio: {phase_df['avg_loss'].mean():.4f}")
        print(f"  Max score: {phase_df['score'].max()}")
        print(f"  Max líneas: {phase_df['lines'].max()}")

    # Análisis de convergencia
    print(f"\n{'='*80}")
    print("ANÁLISIS DE CONVERGENCIA")
    print("="*80)

    # Calcular tendencias (últimos 100 episodios)
    if n_episodes >= 100:
        recent = df.tail(100)

        # Regresión lineal simple para ver tendencias
        x = np.arange(len(recent))

        score_trend = np.polyfit(x, recent['score'], 1)[0]
        lines_trend = np.polyfit(x, recent['lines'], 1)[0]
        loss_trend = np.polyfit(x, recent['avg_loss'], 1)[0]

        print(f"\nTendencias (últimos 100 episodios):")
        print(f"  Score: {'↑ Mejorando' if score_trend > 0 else '↓ Empeorando'} ({score_trend:+.3f}/ep)")
        print(f"  Líneas: {'↑ Mejorando' if lines_trend > 0 else '↓ Empeorando'} ({lines_trend:+.3f}/ep)")
        print(f"  Loss: {'↓ Reduciendo' if loss_trend < 0 else '↑ Aumentando'} ({loss_trend:+.6f}/ep)")

        # Variabilidad
        score_cv = recent['score'].std() / recent['score'].mean() if recent['score'].mean() > 0 else 0
        print(f"\nEstabilidad (coef. variación últimos 100 eps):")
        print(f"  Score CV: {score_cv:.2f} ({'estable' if score_cv < 0.5 else 'variable'})")

    # Análisis de exploración
    print(f"\n{'='*80}")
    print("ANÁLISIS DE EXPLORACIÓN")
    print("="*80)

    epsilon_data = df['epsilon']
    print(f"  Epsilon inicial: {epsilon_data.iloc[0]:.4f}")
    print(f"  Epsilon final: {epsilon_data.iloc[-1]:.4f}")

    # Encontrar cuándo epsilon llegó a epsilon_min
    epsilon_min = config['epsilon_min']
    eps_min_reached = df[df['epsilon'] <= epsilon_min + 0.001]
    if len(eps_min_reached) > 0:
        first_min_ep = eps_min_reached.iloc[0]['episode']
        print(f"  Epsilon mínimo alcanzado en episodio: {int(first_min_ep)}")

    # Gráficos de análisis
    print(f"\n{'='*80}")
    print("Generando gráficos de análisis...")
    print("="*80)

    # Crear directorio de análisis
    analysis_dir = run_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Gráfico 1: Distribución de scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['score'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(df['score'].mean(), color='red', linestyle='--', label=f'Media: {df["score"].mean():.1f}')
    axes[0, 0].axvline(df['score'].median(), color='blue', linestyle='--', label=f'Mediana: {df["score"].median():.1f}')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df['lines'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 1].axvline(df['lines'].mean(), color='red', linestyle='--', label=f'Media: {df["lines"].mean():.1f}')
    axes[0, 1].set_xlabel('Líneas')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Líneas Completadas')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(df['episode'], df['score'], alpha=0.3, s=10, color='green')
    if len(df) >= 50:
        ma50 = df['score'].rolling(window=50).mean()
        axes[1, 0].plot(df['episode'], ma50, color='darkgreen', linewidth=2, label='MA-50')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Score vs Episodio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(df['lines'], df['score'], alpha=0.4, s=20, color='orange')
    axes[1, 1].set_xlabel('Líneas')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Score vs Líneas (correlación)')
    axes[1, 1].grid(True, alpha=0.3)

    # Calcular correlación
    corr = df[['score', 'lines']].corr().iloc[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlación: {corr:.3f}',
                    transform=axes[1, 1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(analysis_dir / "distributions.png", dpi=150)
    plt.close()

    # Gráfico 2: Análisis de learning
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(df['episode'], df['avg_loss'], alpha=0.3, color='red', label='Loss')
    if len(df) >= 50:
        ma50 = df['avg_loss'].rolling(window=50).mean()
        axes[0].plot(df['episode'], ma50, color='darkred', linewidth=2, label='MA-50')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['episode'], df['q_value_avg'], alpha=0.3, color='cyan', label='Q-value')
    if len(df) >= 50:
        ma50 = df['q_value_avg'].rolling(window=50).mean()
        axes[1].plot(df['episode'], ma50, color='darkcyan', linewidth=2, label='MA-50')
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Avg Q-value')
    axes[1].set_title('Average Q-values Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(analysis_dir / "learning_curves.png", dpi=150)
    plt.close()

    # Guardar estadísticas detalladas
    stats = {
        'descriptive_stats': {
            'score': {
                'mean': float(df['score'].mean()),
                'std': float(df['score'].std()),
                'min': int(df['score'].min()),
                'max': int(df['score'].max()),
                'median': float(df['score'].median()),
                'q25': float(df['score'].quantile(0.25)),
                'q75': float(df['score'].quantile(0.75))
            },
            'lines': {
                'mean': float(df['lines'].mean()),
                'std': float(df['lines'].std()),
                'min': int(df['lines'].min()),
                'max': int(df['lines'].max()),
                'median': float(df['lines'].median())
            },
            'reward': {
                'mean': float(df['reward'].mean()),
                'std': float(df['reward'].std()),
                'min': float(df['reward'].min()),
                'max': float(df['reward'].max())
            }
        },
        'training_config': config
    }

    if summary:
        stats['summary'] = summary

    with open(analysis_dir / "detailed_stats.json", 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\n✓ Gráficos guardados en: {analysis_dir}")
    print(f"✓ Estadísticas guardadas en: {analysis_dir / 'detailed_stats.json'}")
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python experiments/analyze_training.py <run_directory>")
        print("\nEjemplo:")
        print("  python experiments/analyze_training.py results/run_20250125_143022")
        print("\nPara analizar el último entrenamiento:")
        print("  python experiments/analyze_training.py results/run_*")
        sys.exit(1)

    run_dir = sys.argv[1]

    # Si se usa wildcard, encontrar el más reciente
    if '*' in run_dir:
        import glob
        runs = glob.glob(run_dir)
        if not runs:
            print(f"No se encontraron directorios que coincidan con: {run_dir}")
            sys.exit(1)
        run_dir = max(runs, key=os.path.getmtime)
        print(f"Analizando el entrenamiento más reciente: {run_dir}\n")

    analyze_training(run_dir)
