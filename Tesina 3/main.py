"""
Analisi di Reti Neurali sul Dataset DARWIN per Diagnosi dell'Alzheimer
=======================================================================
Tesina 3 - MLPClassifier con analisi parametrica completa

Scenari:
  1. Architettura e funzioni di attivazione
  2. Learning rate e ottimizzatori
  3. Regolarizzazione e early stopping

Output per ogni scenario:
  - Curve di apprendimento (train vs validation)
  - Matrici di confusione
  - Curve ROC
  - Box plot accuratezze (cross-validation)
  - Tempi di esecuzione
  - Analisi stabilità

Vincoli:
  - Stesso seed per confronti equi
  - Minimo 30 run per configurazione
  - StandardScaler su dati
  - Cross-validation stratificata
"""

import numpy as np
import pandas as pd
import time
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
SEED        = 42
N_RUNS      = 30
TEST_SIZE   = 0.20
CV_FOLDS    = 5
MAX_ITER    = 1000
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_nn")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# CARICAMENTO E PREPROCESSING
# =============================================================================
def load_darwin(filepath: str):
    """
    Carica DARWIN.csv.
    - Scarta colonna ID (prima)
    - Mappa classe P→1, H→0
    - Imputa missing con mediana
    Restituisce X (numpy), y (numpy), feature_names (list)
    """
    df = pd.read_csv(filepath)
    feature_names = df.columns[1:-1].tolist()
    X = df.iloc[:, 1:-1].copy()
    y = df.iloc[:, -1].map({'P': 1, 'H': 0}).values

    # Imputazione missing values con mediana
    X = X.fillna(X.median())
    X = X.values.astype(float)

    print(f"[Dataset] Shape X: {X.shape} | y: {np.bincount(y)} (H={np.sum(y==0)}, P={np.sum(y==1)})")
    return X, y, feature_names


def make_pipeline(mlp_params: dict) -> Pipeline:
    """Pipeline StandardScaler → MLPClassifier."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',    MLPClassifier(**mlp_params))
    ])


# =============================================================================
# SINGOLO RUN: train/test split + metriche complete
# =============================================================================
def single_run(X, y, mlp_params: dict, run_seed: int) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=run_seed, stratify=y)

    params = {**mlp_params, 'random_state': run_seed}
    pipe = make_pipeline(params)

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    mlp = pipe.named_steps['mlp']
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1] if hasattr(mlp, 'predict_proba') else None

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)

    fpr, tpr, roc_auc = None, None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)

    train_acc = accuracy_score(y_tr, pipe.predict(X_tr))

    return {
        'accuracy':     acc,
        'train_acc':    train_acc,
        'exec_time':    elapsed,
        'loss_curve':   mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else [],
        'n_iter':       mlp.n_iter_,
        'cm':           cm,
        'fpr':          fpr,
        'tpr':          tpr,
        'roc_auc':      roc_auc,
    }


def cv_run(X, y, mlp_params: dict, run_seed: int) -> float:
    """Cross-validation stratificata (5-fold) su tutto il dataset."""
    pipe = make_pipeline({**mlp_params, 'random_state': run_seed})
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=run_seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()


# =============================================================================
# ESPERIMENTO GENERICO: N_RUNS run per configurazione
# =============================================================================
def run_config(X, y, mlp_params: dict, label: str, n_runs: int = N_RUNS) -> dict:
    results = []
    cv_scores = []

    for r in range(n_runs):
        res = single_run(X, y, mlp_params, run_seed=SEED + r)
        results.append(res)
        cv_scores.append(cv_run(X, y, mlp_params, run_seed=SEED + r))
        if (r + 1) % 10 == 0:
            acc_mean = np.mean([x['accuracy'] for x in results])
            print(f"    [{label}] run {r+1}/{n_runs} | acc_mean={acc_mean:.4f}")

    accs      = np.array([r['accuracy']  for r in results])
    train_acc = np.array([r['train_acc'] for r in results])
    times     = np.array([r['exec_time'] for r in results])
    cv_arr    = np.array(cv_scores)

    cm_mean = np.mean([r['cm'] for r in results], axis=0)

    valid_roc = [(i, r) for i, r in enumerate(results) if r['roc_auc'] is not None]
    if valid_roc:
        best_roc_idx = max(valid_roc, key=lambda x: x[1]['roc_auc'])[0]
    else:
        best_roc_idx = 0
    best_fpr = results[best_roc_idx]['fpr']
    best_tpr = results[best_roc_idx]['tpr']
    best_auc = results[best_roc_idx]['roc_auc']

    loss_curves = [r['loss_curve'] for r in results if len(r['loss_curve']) > 0]

    print(f"  → [{label}] acc={accs.mean():.4f}±{accs.std():.4f} "
          f"| cv={cv_arr.mean():.4f}±{cv_arr.std():.4f} "
          f"| time={times.mean():.2f}s")

    return {
        'label':       label,
        'accs':        accs,
        'train_accs':  train_acc,
        'cv_scores':   cv_arr,
        'times':       times,
        'cm_mean':     cm_mean,
        'fpr':         best_fpr,
        'tpr':         best_tpr,
        'roc_auc':     best_auc,
        'loss_curves': loss_curves,
        'acc_mean':    accs.mean(),
        'acc_std':     accs.std(),
        'cv_mean':     cv_arr.mean(),
        'cv_std':      cv_arr.std(),
        'time_mean':   times.mean(),
        'overfitting': float(np.mean(train_acc - accs)),
    }


# =============================================================================
# SCENARI PARAMETRICI
# =============================================================================

def scenario_architecture(X, y) -> dict:
    architectures = {
        '(200,)':         (200,),
        '(400,)':         (400,),
        '(600,)':         (600,),
        '(400,200)':      (400, 200),
        '(600,300)':      (600, 300),
        '(800,400)':      (800, 400),
        '(400,200,100)':  (400, 200, 100),
        '(600,300,150)':  (600, 300, 150),
    }
    activations = ['identity', 'logistic', 'tanh', 'relu']
    results = {}

    for arch_name, arch in architectures.items():
        for act in activations:
            label = f"{arch_name}_{act}"
            print(f"\n[Scenario 1] {label}")
            params = dict(
                hidden_layer_sizes=arch,
                activation=act,
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=MAX_ITER,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
            )
            results[label] = run_config(X, y, params, label)
    return results


def scenario_learning_rate(X, y) -> dict:
    results = {}
    base = dict(hidden_layer_sizes=(400, 200), activation='tanh',
                alpha=0.001, max_iter=MAX_ITER,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=10)

    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for policy in ['constant', 'invscaling', 'adaptive']:
            label = f"lr={lr}_policy={policy}"
            print(f"\n[Scenario 2] {label}")
            params = {**base, 'solver': 'sgd',
                      'learning_rate_init': lr,
                      'learning_rate': policy}
            results[label] = run_config(X, y, params, label)

    for solver in ['adam', 'sgd', 'lbfgs']:
        label = f"solver={solver}"
        print(f"\n[Scenario 2] {label}")
        p = {**base, 'solver': solver, 'learning_rate_init': 0.001}
        if solver == 'lbfgs':
            p.pop('learning_rate', None)
            p.pop('learning_rate_init', None)
            p.update({'early_stopping': False})
        else:
            p['learning_rate'] = 'adaptive'
        results[label] = run_config(X, y, p, label)

    for bs in [16, 32, 64]:
        label = f"batch={bs}"
        print(f"\n[Scenario 2] {label}")
        params = {**base, 'solver': 'adam', 'batch_size': bs,
                  'learning_rate_init': 0.001}
        results[label] = run_config(X, y, params, label)

    return results


def scenario_regularization(X, y) -> dict:
    results = {}
    base = dict(hidden_layer_sizes=(400, 200), activation='tanh',
                solver='adam', learning_rate_init=0.001,
                max_iter=MAX_ITER)

    for alpha in [0.0001, 0.001, 0.01, 0.1, 0.5]:
        label = f"alpha={alpha}"
        print(f"\n[Scenario 3] {label}")
        params = {**base, 'alpha': alpha,
                  'early_stopping': True, 'validation_fraction': 0.15,
                  'n_iter_no_change': 10}
        results[label] = run_config(X, y, params, label)

    for es in [True, False]:
        label = f"early_stop={'on' if es else 'off'}"
        print(f"\n[Scenario 3] {label}")
        params = {**base, 'alpha': 0.001, 'early_stopping': es,
                  'validation_fraction': 0.15, 'n_iter_no_change': 10}
        results[label] = run_config(X, y, params, label)

    for vs in [0.1, 0.15, 0.2]:
        label = f"val_split={vs}"
        print(f"\n[Scenario 3] {label}")
        params = {**base, 'alpha': 0.001, 'early_stopping': True,
                  'validation_fraction': vs, 'n_iter_no_change': 10}
        results[label] = run_config(X, y, params, label)

    for ni in [5, 10, 20]:
        label = f"n_iter_no_change={ni}"
        print(f"\n[Scenario 3] {label}")
        params = {**base, 'alpha': 0.001, 'early_stopping': True,
                  'validation_fraction': 0.15, 'n_iter_no_change': ni}
        results[label] = run_config(X, y, params, label)

    return results


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def _pad(curves):
    max_l = max(len(c) for c in curves)
    return np.array([np.pad(c, (0, max_l - len(c)), mode='edge') for c in curves])


def plot_learning_curves(results: dict, title: str, save_path: str):
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (label, res) in enumerate(results.items()):
        ax = axes[i]
        if res['loss_curves']:
            padded = _pad(res['loss_curves'])
            mean_l = padded.mean(axis=0)
            std_l  = padded.std(axis=0)
            x = np.arange(len(mean_l))
            ax.plot(x, mean_l, color=colors[i % 10], linewidth=1.5)
            ax.fill_between(x, mean_l - std_l, mean_l + std_l,
                            alpha=0.2, color=colors[i % 10])
        ax.set_title(label, fontsize=7, fontweight='bold')
        ax.set_xlabel('Iterazione', fontsize=7)
        ax.set_ylabel('Loss', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_boxplots(results: dict, title: str, save_path: str, key='cv_scores'):
    labels = list(results.keys())
    data   = [results[k][key] for k in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (CV 5-fold)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_confusion_matrices(results: dict, title: str, save_path: str):
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, (label, res) in enumerate(results.items()):
        ax = axes[i]
        cm = res['cm_mean']
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(label, fontsize=7, fontweight='bold')
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks); ax.set_xticklabels(['H', 'P'], fontsize=8)
        ax.set_yticks(tick_marks); ax.set_yticklabels(['H', 'P'], fontsize=8)
        thresh = cm.max() / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f'{cm[row, col]:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if cm[row, col] > thresh else 'black')
        ax.set_xlabel('Predetto', fontsize=7)
        ax.set_ylabel('Reale', fontsize=7)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_roc_curves(results: dict, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (label, res), color in zip(results.items(), colors):
        if res['fpr'] is not None:
            ax.plot(res['fpr'], res['tpr'], color=color, lw=1.5,
                    label=f"{label} (AUC={res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_exec_time(results: dict, title: str, save_path: str):
    labels = list(results.keys())
    data   = [results[k]['times'] for k in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Tempo (s)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def plot_stability(results: dict, title: str, save_path: str):
    labels  = list(results.keys())
    means   = [results[k]['cv_mean'] for k in labels]
    stds    = [results[k]['cv_std']  for k in labels]
    overfits = [results[k]['overfitting'] for k in labels]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.9), 9))

    ax1.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8,
            error_kw=dict(elinewidth=1.5))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax1.set_ylabel('CV Accuracy', fontsize=11)
    ax1.set_title('Accuratezza Media ± Std (stabilità)', fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    colors_of = ['tomato' if v > 0.05 else 'seagreen' for v in overfits]
    ax2.bar(x, overfits, color=colors_of, alpha=0.8)
    ax2.axhline(0.05, color='red', linestyle='--', linewidth=1, label='soglia 5%')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax2.set_ylabel('Gap Train−Test', fontsize=11)
    ax2.set_title('Analisi Overfitting (gap Train−Test)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {save_path}")


def save_summary(results: dict, save_path: str) -> pd.DataFrame:
    rows = []
    for label, res in results.items():
        rows.append({
            'config':        label,
            'acc_mean':      res['acc_mean'],
            'acc_std':       res['acc_std'],
            'cv_mean':       res['cv_mean'],
            'cv_std':        res['cv_std'],
            'overfitting':   res['overfitting'],
            'time_mean_s':   res['time_mean'],
            'roc_auc_best':  res['roc_auc'] if res['roc_auc'] else np.nan,
        })
    df = pd.DataFrame(rows).sort_values('cv_mean', ascending=False)
    df.to_csv(save_path, index=False, float_format='%.5f')
    print(f"  Salvato: {save_path}")
    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  Neural Network Analysis - DARWIN Dataset")
    print("=" * 60)

    # ---- Caricamento --------------------------------------------------------
    X, y, feature_names = load_darwin(r"C:\Users\And11\OneDrive\Desktop\UNIVERSITA\CASSINO\INTELLIGENZA ARTIFICIALE\Gruppo C\Gruppo C\Tesina 3\DARWIN.csv")

    # =========================================================================
    # SCENARIO 1 — Architettura & Attivazione
    # =========================================================================
    print("\n" + "=" * 60)
    print("  SCENARIO 1: Architettura e Funzioni di Attivazione")
    print("=" * 60)

    res_arch = scenario_architecture(X, y)

    plot_learning_curves(res_arch,
        title="Scenario 1 – Curve di Apprendimento (Loss)",
        save_path=f"{OUT_DIR}/s1_learning_curves.png")

    plot_boxplots(res_arch,
        title="Scenario 1 – Distribuzione Accuracy (CV 5-fold)",
        save_path=f"{OUT_DIR}/s1_accuracy_boxplot.png")

    plot_confusion_matrices(res_arch,
        title="Scenario 1 – Matrici di Confusione (media 30 run)",
        save_path=f"{OUT_DIR}/s1_confusion_matrices.png")

    plot_roc_curves(res_arch,
        title="Scenario 1 – Curve ROC (best run per config)",
        save_path=f"{OUT_DIR}/s1_roc_curves.png")

    plot_exec_time(res_arch,
        title="Scenario 1 – Tempi di Esecuzione",
        save_path=f"{OUT_DIR}/s1_exec_time.png")

    plot_stability(res_arch,
        title="Scenario 1 – Stabilità e Overfitting",
        save_path=f"{OUT_DIR}/s1_stability.png")

    s1_df = save_summary(res_arch, f"{OUT_DIR}/s1_summary.csv")
    print("\nTop-5 configurazioni Scenario 1:")
    print(s1_df[['config', 'cv_mean', 'cv_std', 'overfitting', 'time_mean_s']
               ].head(5).to_string(index=False))

    # =========================================================================
    # SCENARIO 2 — Learning Rate & Ottimizzatori
    # =========================================================================
    print("\n" + "=" * 60)
    print("  SCENARIO 2: Learning Rate e Ottimizzatori")
    print("=" * 60)

    res_lr = scenario_learning_rate(X, y)

    plot_learning_curves(res_lr,
        title="Scenario 2 – Curve di Apprendimento (Loss)",
        save_path=f"{OUT_DIR}/s2_learning_curves.png")

    plot_boxplots(res_lr,
        title="Scenario 2 – Distribuzione Accuracy (CV 5-fold)",
        save_path=f"{OUT_DIR}/s2_accuracy_boxplot.png")

    plot_confusion_matrices(res_lr,
        title="Scenario 2 – Matrici di Confusione (media 30 run)",
        save_path=f"{OUT_DIR}/s2_confusion_matrices.png")

    plot_roc_curves(res_lr,
        title="Scenario 2 – Curve ROC",
        save_path=f"{OUT_DIR}/s2_roc_curves.png")

    plot_exec_time(res_lr,
        title="Scenario 2 – Tempi di Esecuzione",
        save_path=f"{OUT_DIR}/s2_exec_time.png")

    plot_stability(res_lr,
        title="Scenario 2 – Stabilità e Overfitting",
        save_path=f"{OUT_DIR}/s2_stability.png")

    s2_df = save_summary(res_lr, f"{OUT_DIR}/s2_summary.csv")
    print("\nTop-5 configurazioni Scenario 2:")
    print(s2_df[['config', 'cv_mean', 'cv_std', 'overfitting', 'time_mean_s']
               ].head(5).to_string(index=False))

    # =========================================================================
    # SCENARIO 3 — Regolarizzazione
    # =========================================================================
    print("\n" + "=" * 60)
    print("  SCENARIO 3: Regolarizzazione")
    print("=" * 60)

    res_reg = scenario_regularization(X, y)

    plot_learning_curves(res_reg,
        title="Scenario 3 – Curve di Apprendimento (Loss)",
        save_path=f"{OUT_DIR}/s3_learning_curves.png")

    plot_boxplots(res_reg,
        title="Scenario 3 – Distribuzione Accuracy (CV 5-fold)",
        save_path=f"{OUT_DIR}/s3_accuracy_boxplot.png")

    plot_confusion_matrices(res_reg,
        title="Scenario 3 – Matrici di Confusione (media 30 run)",
        save_path=f"{OUT_DIR}/s3_confusion_matrices.png")

    plot_roc_curves(res_reg,
        title="Scenario 3 – Curve ROC",
        save_path=f"{OUT_DIR}/s3_roc_curves.png")

    plot_exec_time(res_reg,
        title="Scenario 3 – Tempi di Esecuzione",
        save_path=f"{OUT_DIR}/s3_exec_time.png")

    plot_stability(res_reg,
        title="Scenario 3 – Stabilità e Overfitting",
        save_path=f"{OUT_DIR}/s3_stability.png")

    s3_df = save_summary(res_reg, f"{OUT_DIR}/s3_summary.csv")
    print("\nTop-5 configurazioni Scenario 3:")
    print(s3_df[['config', 'cv_mean', 'cv_std', 'overfitting', 'time_mean_s']
               ].head(5).to_string(index=False))

    # =========================================================================
    # RIEPILOGO GLOBALE
    # =========================================================================
    print("\n" + "=" * 60)
    print("  COMPLETATO — output in:", OUT_DIR)
    print("=" * 60)
    all_results = {**res_arch, **res_lr, **res_reg}
    best = max(all_results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"\n  Migliore configurazione globale: {best[0]}")
    print(f"  CV accuracy: {best[1]['cv_mean']:.4f} ± {best[1]['cv_std']:.4f}")
    print(f"  Overfitting gap: {best[1]['overfitting']:.4f}")