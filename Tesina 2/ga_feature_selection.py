"""
Genetic Algorithm for Feature Selection on DARWIN Dataset
==========================================================

Tesina 2 - Analisi Algoritmo Genetico come Feature Selection
"""

import numpy as np
import pandas as pd
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, List, Dict
from scipy.stats import pointbiserialr

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAZIONE E SEED
# =============================================================================
SEED = 42


# =============================================================================
# CARICAMENTO DATASET
# =============================================================================
def load_darwin_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica il dataset DARWIN e gestisce eventuali missing values.
    - Prima colonna: ID (scartata)
    - Ultime colonna: classe P/H → binarizzata (1=P, 0=H)
    - Colonne intermedie: 450 features
    - Missing values: imputazione con mediana per colonna
    """
    df = pd.read_csv(filepath)
    # Separazione features e target
    X = df.iloc[:, 1:-1].copy()
    y = df.iloc[:, -1].map({'P': 1, 'H': 0})

    # Imputazione missing values con mediana
    X = X.fillna(X.median())

    # Normalizzazione min-max per stabilità numerica
    X = (X - X.min()) / (X.max() - X.min() + 1e-10)

    print(f"[Dataset] Shape: {X.shape}, Classi: {y.value_counts().to_dict()}")
    print(f"[Dataset] Missing values residui: {X.isnull().sum().sum()}")
    return X, y


# =============================================================================
# RAPPRESENTAZIONE INDIVIDUO
# =============================================================================
class Individual:
    """
    Individuo con codifica binaria: 1=feature selezionata, 0=non selezionata.
    Garantisce almeno 1 feature selezionata.
    """

    def __init__(self, n_features: int, chromosome: np.ndarray = None):
        if chromosome is not None:
            self.chromosome = chromosome.copy()
        else:
            self.chromosome = np.random.randint(0, 2, n_features).astype(np.int8)
        # Garantisce almeno 1 feature
        if self.chromosome.sum() == 0:
            self.chromosome[np.random.randint(0, n_features)] = 1
        self.n_features = n_features
        self.fitness = None

    def count_selected_features(self) -> int:
        return int(self.chromosome.sum())

    def get_selected_indices(self) -> np.ndarray:
        return np.where(self.chromosome == 1)[0]


# =============================================================================
# FUNZIONE FITNESS (CFS - Correlation-based Feature Selection)
# =============================================================================
# Cache globale per correlazioni (velocizza i run ripetuti)
_corr_cache: Dict[str, float] = {}


def _point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Correlazione punto-biseriale |r| tra feature continua e classe binaria."""
    key = id(x)
    if key in _corr_cache:
        return _corr_cache[key]
    try:
        r, _ = pointbiserialr(x, y)
        val = abs(r) if not np.isnan(r) else 0.0
    except Exception:
        val = 0.0
    _corr_cache[key] = val
    return val


def fitness_correlation_based(individual: Individual,
                               X: pd.DataFrame,
                               y: pd.Series,
                               rcf_matrix: np.ndarray = None,
                               rfc_vector: np.ndarray = None) -> float:
    """
    CFS (Hall, 1999):
        Merit_k = (k * r_cf) / sqrt(k + k*(k-1) * r_ff)

    dove:
        k      = numero di feature selezionate
        r_cf   = media delle |correlazioni feature-classe|
        r_ff   = media delle |correlazioni inter-feature|

    Parametri pre-calcolati (rcf_matrix, rfc_vector) per efficienza.
    """
    selected = individual.get_selected_indices()
    k = len(selected)
    if k == 0:
        return 0.0

    # Feature-class correlation (media)
    r_cf = float(np.mean(rfc_vector[selected]))

    # Feature-feature correlation (media triangolo superiore)
    if k == 1:
        r_ff = 0.0
    else:
        sub = rcf_matrix[np.ix_(selected, selected)]
        upper = sub[np.triu_indices(k, k=1)]
        r_ff = float(np.mean(np.abs(upper))) if len(upper) > 0 else 0.0

    denominator = np.sqrt(k + k * (k - 1) * r_ff)
    if denominator < 1e-10:
        return 0.0

    merit = (k * r_cf) / denominator
    return float(merit)


def precompute_correlations(X: pd.DataFrame, y: pd.Series):
    """Pre-calcola matrici di correlazione per efficienza."""
    n = X.shape[1]
    Xv = X.values
    yv = y.values

    # Feature-class
    rfc = np.array([abs(np.corrcoef(Xv[:, i], yv)[0, 1])
                    if not np.isnan(np.corrcoef(Xv[:, i], yv)[0, 1]) else 0.0
                    for i in range(n)])

    # Feature-feature (matrice simmetrica delle correlazioni assolute)
    rcf_matrix = X.corr().abs().values.copy()
    np.fill_diagonal(rcf_matrix, 0.0)

    print(f"[Correlations] r_fc mean={rfc.mean():.4f}, max={rfc.max():.4f}")
    return rcf_matrix, rfc


# =============================================================================
# OPERATORI GENETICI
# =============================================================================

def tournament_selection(population: List[Individual],
                         fitness_values: List[float],
                         tournament_size: int = 3) -> Individual:
    """Selezione a torneo: k individui random, vince il migliore."""
    idx = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = idx[np.argmax([fitness_values[i] for i in idx])]
    return population[best_idx]


def roulette_wheel_selection(population: List[Individual],
                             fitness_values: List[float]) -> Individual:
    """Roulette wheel: probabilità proporzionale al fitness (shiftato ≥ 0)."""
    fv = np.array(fitness_values, dtype=float)
    fv -= fv.min()  # shift per evitare valori negativi
    total = fv.sum()
    if total < 1e-10:
        return population[np.random.randint(len(population))]
    probs = fv / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def single_point_crossover(parent1: Individual,
                           parent2: Individual,
                           crossover_rate: float = 0.8) -> Tuple[Individual, Individual]:
    """Crossover a singolo punto con probabilità crossover_rate."""
    n = parent1.n_features
    if np.random.random() < crossover_rate:
        point = np.random.randint(1, n)
        c1 = np.concatenate([parent1.chromosome[:point], parent2.chromosome[point:]])
        c2 = np.concatenate([parent2.chromosome[:point], parent1.chromosome[point:]])
        return Individual(n, c1), Individual(n, c2)
    else:
        return Individual(n, parent1.chromosome), Individual(n, parent2.chromosome)


def bit_flip_mutation(individual: Individual,
                      mutation_rate: float = 0.1) -> Individual:
    """Mutazione bit-flip: ogni bit viene invertito con prob mutation_rate."""
    chrom = individual.chromosome.copy()
    mask = np.random.random(len(chrom)) < mutation_rate
    chrom[mask] ^= 1
    return Individual(individual.n_features, chrom)


# =============================================================================
# ALGORITMO GENETICO
# =============================================================================
class GeneticAlgorithm:
    """Algoritmo Genetico modulare per feature selection."""

    def __init__(self,
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 tournament_size: int = 3,
                 max_generations: int = 100,
                 convergence_threshold: int = None,
                 convergence_tolerance: float = 1e-5,
                 random_seed: int = SEED):

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.convergence_tolerance = convergence_tolerance
        self.random_seed = random_seed

    def _select(self, population, fitness_values):
        if self.selection_method == 'roulette':
            return roulette_wheel_selection(population, fitness_values)
        else:  # tournament (default)
            return tournament_selection(population, fitness_values, self.tournament_size)

    def initialize_population(self, n_features: int) -> List[Individual]:
        np.random.seed(self.random_seed)
        return [Individual(n_features) for _ in range(self.population_size)]

    def evaluate_population(self, population: List[Individual],
                            X: pd.DataFrame, y: pd.Series,
                            rcf_matrix=None, rfc_vector=None) -> List[float]:
        return [fitness_correlation_based(ind, X, y, rcf_matrix, rfc_vector)
                for ind in population]

    def calculate_population_diversity(self, population: List[Individual]) -> float:
        """Diversità = varianza media dei bit nei cromosomi."""
        chroms = np.array([ind.chromosome for ind in population], dtype=float)
        return float(np.mean(np.var(chroms, axis=0)))

    def run(self, X: pd.DataFrame, y: pd.Series,
            rcf_matrix=None, rfc_vector=None) -> Dict:
        """Esegue il GA e restituisce dizionario con tutte le metriche."""
        np.random.seed(self.random_seed)
        n_features = X.shape[1]
        start_time = time.time()

        population = self.initialize_population(n_features)
        fitness_values = self.evaluate_population(population, X, y, rcf_matrix, rfc_vector)

        best_idx = int(np.argmax(fitness_values))
        best_individual = Individual(n_features, population[best_idx].chromosome)
        best_fitness = fitness_values[best_idx]

        # Log per generazione
        gen_best_fitness = []
        gen_avg_fitness = []
        gen_diversity = []

        no_improve_count = 0
        generation = 0

        for gen in range(self.max_generations):
            generation = gen

            # Elitismo: conserva il miglior individuo
            new_population = [Individual(n_features, best_individual.chromosome)]

            # Generazione nuova popolazione
            while len(new_population) < self.population_size:
                p1 = self._select(population, fitness_values)
                p2 = self._select(population, fitness_values)
                c1, c2 = single_point_crossover(p1, p2, self.crossover_rate)
                c1 = bit_flip_mutation(c1, self.mutation_rate)
                c2 = bit_flip_mutation(c2, self.mutation_rate)
                new_population.extend([c1, c2])

            population = new_population[:self.population_size]
            fitness_values = self.evaluate_population(population, X, y, rcf_matrix, rfc_vector)

            gen_best = max(fitness_values)
            gen_avg = float(np.mean(fitness_values))
            diversity = self.calculate_population_diversity(population)

            gen_best_fitness.append(gen_best)
            gen_avg_fitness.append(gen_avg)
            gen_diversity.append(diversity)

            # Aggiorna best globale
            if gen_best > best_fitness + self.convergence_tolerance:
                best_idx = int(np.argmax(fitness_values))
                best_individual = Individual(n_features, population[best_idx].chromosome)
                best_fitness = gen_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Criterio di convergenza
            if self.convergence_threshold is not None:
                if no_improve_count >= self.convergence_threshold:
                    break

        elapsed = time.time() - start_time

        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'generations_completed': generation + 1,
            'execution_time': elapsed,
            'gen_best_fitness': gen_best_fitness,
            'gen_avg_fitness': gen_avg_fitness,
            'gen_diversity': gen_diversity,
            'n_selected_features': best_individual.count_selected_features(),
            'selected_features': best_individual.get_selected_indices(),
        }


# =============================================================================
# LOGGING E METRICHE
# =============================================================================
class ExperimentLogger:
    """Raccoglie e aggrega i risultati di più run."""

    def __init__(self):
        self.runs = []
        self.feature_counts = {}

    def log_run(self, run_id: int, result: Dict, config: Dict):
        entry = {
            'run_id': run_id,
            'best_fitness': result['best_fitness'],
            'n_selected': result['n_selected_features'],
            'exec_time': result['execution_time'],
            'generations': result['generations_completed'],
            'gen_best': result['gen_best_fitness'],
            'gen_avg': result['gen_avg_fitness'],
            'gen_diversity': result['gen_diversity'],
            **config
        }
        self.runs.append(entry)
        for fi in result['selected_features']:
            self.feature_counts[fi] = self.feature_counts.get(fi, 0) + 1

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ('gen_best', 'gen_avg', 'gen_diversity')}
                              for r in self.runs])


# =============================================================================
# ESPERIMENTI PARAMETRICI
# =============================================================================

def _run_config(ga_kwargs: Dict, X, y, rcf_matrix, rfc_vector,
                n_runs: int, config_label: str) -> Dict:
    """Helper: esegue n_runs con una configurazione GA e restituisce statistiche."""
    logger = ExperimentLogger()
    for run in range(n_runs):
        ga = GeneticAlgorithm(**ga_kwargs, random_seed=SEED + run)
        result = ga.run(X, y, rcf_matrix, rfc_vector)
        logger.log_run(run, result, {'config': config_label})
        if (run + 1) % 5 == 0:
            print(f"  [{config_label}] run {run+1}/{n_runs} "
                  f"| best_fit={result['best_fitness']:.4f} "
                  f"| n_feat={result['n_selected_features']} "
                  f"| time={result['execution_time']:.2f}s")
    df = logger.summary()
    return {
        'label': config_label,
        'logger': logger,
        'df': df,
        'best_fitness_mean': df['best_fitness'].mean(),
        'best_fitness_std': df['best_fitness'].std(),
        'exec_time_mean': df['exec_time'].mean(),
        'n_selected_mean': df['n_selected'].mean(),
        'gen_completed_mean': df['generations'].mean(),
        'convergence_curves': [r['gen_best'] for r in logger.runs],
    }


def run_experiment_population_size(X, y, rcf_matrix, rfc_vector, n_runs: int = 30) -> Dict:
    """Scenario 1: Test dimensioni popolazione [20, 50, 100, 200, 500]."""
    population_sizes = [20, 50, 100, 200, 500]
    results = {}
    for ps in population_sizes:
        print(f"\n[Scenario 1] Population size = {ps}")
        ga_kwargs = dict(population_size=ps, crossover_rate=0.8, mutation_rate=0.1,
                         selection_method='tournament', tournament_size=3,
                         max_generations=100)
        results[ps] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector,
                                  n_runs, f"pop={ps}")
    return results


def run_experiment_genetic_operators(X, y, rcf_matrix, rfc_vector, n_runs: int = 30) -> Dict:
    """Scenario 2: Crossover, mutation, selection method."""
    results = {}

    # Crossover rates
    for cr in [0.6, 0.7, 0.8, 0.9]:
        label = f"cr={cr}"
        print(f"\n[Scenario 2] Crossover rate = {cr}")
        ga_kwargs = dict(population_size=100, crossover_rate=cr, mutation_rate=0.1,
                         selection_method='tournament', tournament_size=3,
                         max_generations=100)
        results[label] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector, n_runs, label)

    # Mutation rates
    for mr in [0.01, 0.05, 0.1, 0.15]:
        label = f"mr={mr}"
        print(f"\n[Scenario 2] Mutation rate = {mr}")
        ga_kwargs = dict(population_size=100, crossover_rate=0.8, mutation_rate=mr,
                         selection_method='tournament', tournament_size=3,
                         max_generations=100)
        results[label] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector, n_runs, label)

    # Selection methods: Tournament k=2,3,4 + Roulette
    for k in [2, 3, 4]:
        label = f"tournament_k={k}"
        print(f"\n[Scenario 2] Tournament k={k}")
        ga_kwargs = dict(population_size=100, crossover_rate=0.8, mutation_rate=0.1,
                         selection_method='tournament', tournament_size=k,
                         max_generations=100)
        results[label] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector, n_runs, label)

    print("\n[Scenario 2] Roulette Wheel")
    ga_kwargs = dict(population_size=100, crossover_rate=0.8, mutation_rate=0.1,
                     selection_method='roulette', max_generations=100)
    results['roulette'] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector,
                                      n_runs, 'roulette')

    return results


def run_experiment_stopping_criteria(X, y, rcf_matrix, rfc_vector, n_runs: int = 30) -> Dict:
    """Scenario 3: Criteri di stop."""
    results = {}

    # Generazioni fisse
    for mg in [50, 100, 200]:
        label = f"maxgen={mg}"
        print(f"\n[Scenario 3] Max generations = {mg}")
        ga_kwargs = dict(population_size=100, crossover_rate=0.8, mutation_rate=0.1,
                         selection_method='tournament', tournament_size=3,
                         max_generations=mg)
        results[label] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector, n_runs, label)

    # Convergenza: soglie × tolleranze
    for thresh in [10, 20, 30]:
        for tol in [1e-4, 1e-5, 1e-6]:
            label = f"conv_t={thresh}_tol={tol:.0e}"
            print(f"\n[Scenario 3] Convergence thresh={thresh}, tol={tol}")
            ga_kwargs = dict(population_size=100, crossover_rate=0.8, mutation_rate=0.1,
                             selection_method='tournament', tournament_size=3,
                             max_generations=200,
                             convergence_threshold=thresh,
                             convergence_tolerance=tol)
            results[label] = _run_config(ga_kwargs, X, y, rcf_matrix, rfc_vector, n_runs, label)

    return results


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def _pad_curves(curves):
    """Porta tutte le curve alla stessa lunghezza (padding con ultimo valore)."""
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        arr = np.array(c, dtype=float)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
        padded.append(arr)
    return np.array(padded)


def plot_convergence_curves(results: Dict, title: str = "Convergence Curves",
                            save_path: str = None):
    """Curve di convergenza: media ± std su tutti i run."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (label, res), color in zip(results.items(), colors):
        curves = res['convergence_curves']
        padded = _pad_curves(curves)
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, label=label, color=color, linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=color)

    ax.set_xlabel('Generazione', fontsize=12)
    ax.set_ylabel('Best Fitness (CFS Merit)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_fitness_boxplots(results: Dict, title: str = "Fitness Distribution",
                          save_path: str = None):
    """Box plot del best fitness per configurazione."""
    labels = list(results.keys())
    data = [results[k]['df']['best_fitness'].values for k in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best Fitness (CFS Merit)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_feature_frequency(feature_counts: Dict, feature_names: List[str],
                           top_k: int = 30, title: str = "Feature Selection Frequency",
                           save_path: str = None):
    """Istogramma delle top-k features più frequentemente selezionate."""
    if not feature_counts:
        return
    sorted_items = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    indices, counts = zip(*sorted_items)
    names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), counts, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Frequenza di Selezione (su 30 run)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def save_summary_table(results: Dict, scenario_name: str, save_path: str):
    """Salva tabella riassuntiva CSV con metriche aggregate."""
    rows = []
    for label, res in results.items():
        df = res['df']
        rows.append({
            'config': label,
            'best_fitness_mean': df['best_fitness'].mean(),
            'best_fitness_std': df['best_fitness'].std(),
            'best_fitness_median': df['best_fitness'].median(),
            'n_selected_mean': df['n_selected'].mean(),
            'n_selected_std': df['n_selected'].std(),
            'exec_time_mean': df['exec_time'].mean(),
            'exec_time_std': df['exec_time'].std(),
            'gen_completed_mean': df['generations'].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(save_path, index=False, float_format='%.6f')
    print(f"  Salvato: {save_path}")
    return summary


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import os
    import sys

    OUT_DIR = "results"
    os.makedirs(OUT_DIR, exist_ok=True)

    N_RUNS = 30  # minimo richiesto dalla tesina

    print("=" * 60)
    print("  GA Feature Selection - DARWIN Dataset")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Caricamento dati
    # -------------------------------------------------------------------------
    X, y = load_darwin_dataset(r"C:\Users\And11\OneDrive\Desktop\UNIVERSITA\CASSINO\INTELLIGENZA ARTIFICIALE\Gruppo C\Gruppo C\Tesina 2\DARWIN.csv")    
    feature_names = list(X.columns)
    # Pre-calcolo correlazioni (una sola volta, riutilizzato in tutti gli esperimenti)
    print("\n[Pre-computation] Calcolo correlazioni...")
    rcf_matrix, rfc_vector = precompute_correlations(X, y)

    # -------------------------------------------------------------------------
    # 2. Scenario 1 – Dimensione della Popolazione
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SCENARIO 1: Dimensione della Popolazione")
    print("=" * 60)
    res_pop = run_experiment_population_size(X, y, rcf_matrix, rfc_vector, N_RUNS)

    plot_convergence_curves(res_pop,
                            title="Scenario 1 – Convergenza per Dimensione Popolazione",
                            save_path=f"{OUT_DIR}/s1_convergence.png")
    plot_fitness_boxplots(res_pop,
                          title="Scenario 1 – Distribuzione Fitness (Best) per Dimensione Popolazione",
                          save_path=f"{OUT_DIR}/s1_fitness_boxplot.png")

    # Feature frequency per scenario 1 (pop=100)
    for ps, res in res_pop.items():
        plot_feature_frequency(
            res['logger'].feature_counts, 
            feature_names,
            title=f"Scenario 1 (pop={ps}) – Frequenza Selezione Feature",
            save_path=f"{OUT_DIR}/s1_feature_freq_pop_{ps}.png"
        )
    s1_summary = save_summary_table(res_pop, "Population Size",
                                    f"{OUT_DIR}/s1_summary.csv")
    print("\nScenario 1 – Riepilogo:")
    print(s1_summary[['config', 'best_fitness_mean', 'best_fitness_std',
                       'exec_time_mean', 'n_selected_mean']].to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. Scenario 2 – Operatori Genetici
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SCENARIO 2: Operatori Genetici")
    print("=" * 60)
    res_ops = run_experiment_genetic_operators(X, y, rcf_matrix, rfc_vector, N_RUNS)

    plot_convergence_curves(res_ops,
                            title="Scenario 2 – Convergenza per Operatori Genetici",
                            save_path=f"{OUT_DIR}/s2_convergence.png")
    plot_fitness_boxplots(res_ops,
                          title="Scenario 2 – Distribuzione Fitness per Operatori Genetici",
                          save_path=f"{OUT_DIR}/s2_fitness_boxplot.png")

    for label, res in res_ops.items():
        # Rendiamo il label adatto a un nome file (es. cr=0.8 -> cr_08)
        clean_label = label.replace("=", "_").replace(".", "")
        plot_feature_frequency(
            res['logger'].feature_counts, 
            feature_names,
            title=f"Scenario 2 ({label}) – Frequenza Selezione Feature",
            save_path=f"{OUT_DIR}/s2_feature_freq_{clean_label}.png"
        )

    s2_summary = save_summary_table(res_ops, "Genetic Operators",
                                    f"{OUT_DIR}/s2_summary.csv")
    print("\nScenario 2 – Riepilogo:")
    print(s2_summary[['config', 'best_fitness_mean', 'best_fitness_std',
                       'exec_time_mean', 'n_selected_mean']].to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Scenario 3 – Criteri di Stop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SCENARIO 3: Criteri di Stop")
    print("=" * 60)
    res_stop = run_experiment_stopping_criteria(X, y, rcf_matrix, rfc_vector, N_RUNS)

    plot_convergence_curves(res_stop,
                            title="Scenario 3 – Convergenza per Criteri di Stop",
                            save_path=f"{OUT_DIR}/s3_convergence.png")
    plot_fitness_boxplots(res_stop,
                          title="Scenario 3 – Distribuzione Fitness per Criteri di Stop",
                          save_path=f"{OUT_DIR}/s3_fitness_boxplot.png")
    for label, res in res_stop.items():
        clean_label = label.replace("=", "_").replace(".", "").replace(":", "_")
        plot_feature_frequency(
            res['logger'].feature_counts, 
            feature_names,
            title=f"Scenario 3 ({label}) – Frequenza Selezione Feature",
            save_path=f"{OUT_DIR}/s3_feature_freq_{clean_label}.png"
        )
        
    s3_summary = save_summary_table(res_stop, "Stopping Criteria",
                                    f"{OUT_DIR}/s3_summary.csv")
    print("\nScenario 3 – Riepilogo:")
    print(s3_summary[['config', 'best_fitness_mean', 'best_fitness_std',
                       'exec_time_mean', 'gen_completed_mean']].to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Riepilogo finale
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPLETATO")
    print(f"  Output salvati in: {OUT_DIR}/")
    print("=" * 60)