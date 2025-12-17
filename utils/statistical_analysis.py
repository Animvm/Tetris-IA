import numpy as np
from scipy import stats

# calcula metricas de resumen del entrenamiento
def compute_summary_metrics(metrics_dict):
    n_final = min(100, len(metrics_dict['scores']))

    return {
        'final_score_mean': np.mean(metrics_dict['scores'][-n_final:]),
        'final_score_std': np.std(metrics_dict['scores'][-n_final:]),
        'final_score_median': np.median(metrics_dict['scores'][-n_final:]),
        'final_lines_mean': np.mean(metrics_dict['lines'][-n_final:]),
        'final_lines_std': np.std(metrics_dict['lines'][-n_final:]),
        'final_lines_median': np.median(metrics_dict['lines'][-n_final:]),
        'final_pieces_mean': np.mean(metrics_dict['pieces'][-n_final:]),

        'max_score': np.max(metrics_dict['scores']),
        'max_lines': np.max(metrics_dict['lines']),
        'max_pieces': np.max(metrics_dict['pieces']),

        'avg_time_per_episode': np.mean(metrics_dict['computation_time']),
        'std_time_per_episode': np.std(metrics_dict['computation_time']),
        'total_time': np.sum(metrics_dict['computation_time']),

        'score_auc': np.trapz(metrics_dict['scores']),
        'lines_auc': np.trapz(metrics_dict['lines']),

        'score_variance': np.var(metrics_dict['scores'][-n_final:]),
        'lines_variance': np.var(metrics_dict['lines'][-n_final:]),

        'total_episodes': len(metrics_dict['scores'])
    }

# compara estadisticamente diferentes agentes
def statistical_comparison(results_dict):
    agents = list(results_dict.keys())
    comparisons = []

    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            agent1, agent2 = agents[i], agents[j]

            n = min(100, min(len(results_dict[agent1]['scores']),
                            len(results_dict[agent2]['scores'])))

            scores1 = results_dict[agent1]['scores'][-n:]
            scores2 = results_dict[agent2]['scores'][-n:]

            # test de Mann-Whitney U
            statistic, p_value = stats.mannwhitneyu(
                scores1, scores2, alternative='two-sided'
            )

            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            std1, std2 = np.std(scores1), np.std(scores2)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

            if abs(cohens_d) < 0.2:
                effect_magnitude = "pequeno"
            elif abs(cohens_d) < 0.5:
                effect_magnitude = "mediano"
            elif abs(cohens_d) < 0.8:
                effect_magnitude = "grande"
            else:
                effect_magnitude = "muy grande"

            comparisons.append({
                'agent1': agent1,
                'agent2': agent2,
                'mean1': mean1,
                'mean2': mean2,
                'std1': std1,
                'std2': std2,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_magnitude': effect_magnitude,
                'significant': p_value < 0.05,
                'winner': agent1 if mean1 > mean2 else agent2,
                'difference_pct': abs(mean1 - mean2) / max(mean1, mean2) * 100
            })

    return comparisons

def print_statistical_report(results_dict):
    print("="*70)
    print("ANALISIS ESTADISTICO")
    print("="*70)

    print("\nRESUMEN POR AGENTE:")
    print("-"*70)

    for name, metrics in results_dict.items():
        summary = compute_summary_metrics(metrics)
        print(f"\n{name.upper()}:")
        print(f"  Score final: {summary['final_score_mean']:.1f} +/- {summary['final_score_std']:.1f}")
        print(f"  Lineas final: {summary['final_lines_mean']:.1f} +/- {summary['final_lines_std']:.1f}")
        print(f"  Score maximo: {summary['max_score']:.1f}")
        print(f"  Lineas maximo: {summary['max_lines']}")
        print(f"  Tiempo/episodio: {summary['avg_time_per_episode']:.2f}s")
        print(f"  Sample efficiency (AUC): {summary['score_auc']:.0f}")

    print("\n" + "="*70)
    print("COMPARACIONES PAREADAS:")
    print("-"*70)

    comparisons = statistical_comparison(results_dict)

    for comp in comparisons:
        print(f"\n{comp['agent1'].upper()} vs {comp['agent2'].upper()}:")
        print(f"  Medias: {comp['mean1']:.1f} vs {comp['mean2']:.1f}")
        print(f"  Ganador: {comp['winner'].upper()} (diferencia: {comp['difference_pct']:.1f}%)")
        print(f"  p-value: {comp['p_value']:.4f} "
              f"{'(significativo)' if comp['significant'] else '(no significativo)'}")
        print(f"  Cohen's d: {comp['cohens_d']:.3f} (efecto {comp['effect_magnitude']})")

    print("\n" + "="*70)

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - margin, mean + margin
