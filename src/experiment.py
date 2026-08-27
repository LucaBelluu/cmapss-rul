"""Motore di esperimento: dalla specifica di un modello agli artefatti del confronto.

Ruolo nel progetto
    Compone i due stadi previsti dal protocollo (ricerca della configurazione
    sulle partizioni del seme dedicato, rivalutazione della sola configurazione
    selezionata su tutte le partizioni di confronto) ed e' l'unico punto in cui
    questa composizione e' scritta. Ogni blocco del confronto passa da qui:
    modelli valutati sotto procedure diverse non sarebbero confrontabili, e la
    parita' di trattamento va garantita dal codice e non dalla disciplina di
    chi lo usa.

Cosa riceve
    Una specifica di modello del registro, oppure il nome di un metodo di
    selezione delle variabili, insieme alla matrice di progetto di un
    sottoinsieme.

Cosa produce
    Una struttura `ModelRun` con la configurazione selezionata, le metriche per
    fold, il riepilogo su media e dispersione, i parametri leggibili del
    modello e, per i modelli con iperparametri, la griglia completa. Non scrive
    su disco: la persistenza e' compito degli script di orchestrazione, come
    per il protocollo.

I due stadi
    La ricerca opera sulle 5 partizioni del seme di ricerca, un solo passaggio.
    La configurazione selezionata viene poi rivalutata sulle 15 partizioni dei
    tre semi di confronto, e sono quei 15 punteggi a produrre la media e la
    dispersione riportate in tabella. La cross-validation non e' annidata: il
    punteggio riportato e' ottimisticamente distorto, e la stima non
    condizionata dalla selezione proviene dall'insieme di verifica ufficiale,
    che non viene letto qui.

Baseline
    Le due baseline sono costruite qui e non nei singoli script, per la stessa
    ragione per cui i due stadi sono scritti in un punto solo: ogni blocco del
    confronto ne ha bisogno per rendere leggibile la propria tabella, e due
    definizioni separate potrebbero divergere senza che nulla lo segnali.

Selezione delle variabili
    I tre metodi di selezione sono trattati come gli altri modelli: la ricerca
    del sottoinsieme avviene sulle partizioni del seme di ricerca, il
    sottoinsieme selezionato diventa una pipeline a colonne fisse, e questa
    viene rivalutata sulle partizioni di confronto. La distorsione ottimistica
    di questo trattamento cresce con il numero di configurazioni esplorate, ed
    e' quindi maggiore per la ricerca esaustiva che per una griglia di
    cinquanta valori. `nested_selection_check` la misura, rifacendo la
    selezione dentro ciascun fold di confronto: il divario fra i due punteggi
    e' la stima diretta dell'ottimismo introdotto dalla selezione. Quel
    risultato e' diagnostico e non entra in graduatoria.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.pipeline import build_pipeline
from src.protocol import COMPARISON_SEEDS, N_SPLITS, evaluate, make_splits, summarize
from src.registry import ModelSpec
from src.search import SearchResult, grid_search
from src.selection import SELECTION_METHODS, build_grams, _fold_rmse, pick_best


@dataclass
class ModelRun:
    """Esito completo dell'esperimento su un modello.

    estimator
        La pipeline con la configurazione selezionata, non adattata. E' l'
        oggetto da riaddestrare sull'intera parte di addestramento quando la
        graduatoria verra' chiusa e l'insieme di verifica ufficiale letto.
    """

    key: str
    label: str
    subset: str
    estimator: object = field(repr=False)
    config: dict = field(default_factory=dict)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    summary: pd.Series = field(default_factory=pd.Series, repr=False)
    grid_table: pd.DataFrame | None = field(default=None, repr=False)
    coefficients: pd.DataFrame | None = field(default=None, repr=False)
    selection_history: pd.DataFrame | None = field(default=None, repr=False)
    diagnostics: dict = field(default_factory=dict)


def _config_label(config: dict) -> str:
    """Configurazione selezionata in forma leggibile per le tabelle."""
    if not config:
        return "nessun iperparametro"
    parts = []
    for name, value in config.items():
        short = name.split("__")[-1]
        parts.append(f"{short}={value:.4g}" if isinstance(value, float) else f"{short}={value}")
    return ", ".join(parts)


def baseline_runs(design, comparison_splits) -> list[ModelRun]:
    """Le due baseline, valutate sulle stesse partizioni degli altri modelli.

    Sono ricalcolate a ogni blocco e non riprese dagli artefatti di un altro:
    la tabella di ciascun blocco e' cosi' prodotta interamente da una sola
    esecuzione, e non dalla composizione di esecuzioni diverse. Il costo e'
    trascurabile e il guadagno e' che ogni tabella si legge da sola.

    La predizione costante e' il pavimento assoluto, la regressione sul solo
    numero di ciclo e' il pavimento informativo: il guadagno di un modello si
    legge rispetto alla seconda.
    """
    from src.baselines import all_baselines

    runs = []
    labels = {
        "baseline_costante": "Predizione costante",
        "baseline_solo_ciclo": "Regressione sul solo numero di ciclo",
    }
    for key, model in all_baselines().items():
        fold_metrics = evaluate(
            model, design.X_train, design.y_train, design.groups_train, comparison_splits
        )
        fold_metrics.insert(0, "model", key)
        summary = summarize(fold_metrics, label=key)
        summary["label"] = labels[key]
        summary["subset"] = design.subset
        summary["config"] = "baseline"
        summary["search_seconds"] = 0.0
        summary["n_configurations"] = 1
        summary["n_nonzero"] = 0 if key == "baseline_costante" else 1
        runs.append(
            ModelRun(
                key=key,
                label=labels[key],
                subset=design.subset,
                estimator=model,
                fold_metrics=fold_metrics,
                summary=summary,
            )
        )
    return runs


def run_grid_model(
    spec: ModelSpec,
    design,
    *,
    search_splits,
    comparison_splits,
    n_jobs: int = -1,
) -> ModelRun:
    """Esegue i due stadi su un modello con griglia di iperparametri."""
    X, y, groups = design.X_train, design.y_train, design.groups_train
    features = list(design.features)
    param_grid = spec.param_grid(len(features))
    pipeline = build_pipeline(clone(spec.estimator))

    search: SearchResult | None = None
    if param_grid:
        search = grid_search(pipeline, param_grid, X, y, search_splits, n_jobs=n_jobs)
        selected = clone(pipeline).set_params(**search.best_params)
        config = dict(search.best_params)
    else:
        selected = clone(pipeline)
        config = {}

    fold_metrics = evaluate(selected, X, y, groups, comparison_splits)
    fold_metrics.insert(0, "model", spec.key)

    summary = summarize(fold_metrics, label=spec.key)
    summary["label"] = spec.label
    summary["subset"] = design.subset
    summary["config"] = _config_label(config)

    # I parametri leggibili si estraggono dal modello riaddestrato sull'intera
    # parte di addestramento. Non usa in alcun modo l'insieme di verifica, ed e'
    # il modello di cui si commenta il comportamento: uno dei quindici modelli
    # di fold sarebbe una scelta arbitraria fra quindici stime diverse.
    fitted = clone(selected).fit(X, np.asarray(y))
    coefficients = spec.reader(fitted, features) if spec.reader is not None else None
    if coefficients is not None:
        coefficients.insert(0, "model", spec.key)

    diagnostics = {"note": spec.note}
    if search is not None:
        summary["search_seconds"] = search.seconds
        summary["n_configurations"] = len(search.table)
        diagnostics.update(
            {
                "search_seconds": search.seconds,
                "n_configurations": len(search.table),
                "boundary": ", ".join(search.boundary),
                "convergence_warnings": search.convergence_warnings,
                "failed_configurations": search.failed_configurations,
                "search_best_rmse": search.best_score,
            }
        )
        summary["n_nonzero"] = _count_nonzero(coefficients)
    else:
        summary["search_seconds"] = 0.0
        summary["n_configurations"] = 1
        summary["n_nonzero"] = _count_nonzero(coefficients)

    return ModelRun(
        key=spec.key,
        label=spec.label,
        subset=design.subset,
        estimator=selected,
        config=config,
        fold_metrics=fold_metrics,
        summary=summary,
        grid_table=search.table if search is not None else None,
        coefficients=coefficients,
        diagnostics=diagnostics,
    )


def run_selection_model(
    method: str,
    label: str,
    design,
    *,
    search_splits,
    comparison_splits,
) -> ModelRun:
    """Esegue i due stadi su un modello che passa dalla selezione delle variabili."""
    X, y, groups = design.X_train, design.y_train, design.groups_train
    features = list(design.features)

    start = time.perf_counter()
    history = SELECTION_METHODS[method](X.to_numpy(), np.asarray(y), search_splits, features)
    seconds = time.perf_counter() - start

    best = pick_best(history)
    selected_features = best["selected_features"]

    from sklearn.linear_model import LinearRegression

    selected = build_pipeline(LinearRegression(), columns=selected_features)
    fold_metrics = evaluate(selected, X, y, groups, comparison_splits)
    fold_metrics.insert(0, "model", method)

    summary = summarize(fold_metrics, label=method)
    summary["label"] = label
    summary["subset"] = design.subset
    summary["config"] = f"k={best['k']}"
    summary["search_seconds"] = seconds
    summary["n_configurations"] = int(history["n_subsets"].sum()) if "n_subsets" in history else len(history)
    summary["n_nonzero"] = best["k"]

    fitted = clone(selected).fit(X, np.asarray(y))
    coefficients = pd.DataFrame(
        {
            "model": method,
            "feature": selected_features,
            "coef": np.asarray(fitted.named_steps["model"].coef_).ravel(),
        }
    )
    coefficients["abs_coef"] = coefficients["coef"].abs()
    coefficients["zero"] = False
    coefficients = coefficients.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    history_out = history.copy()
    history_out.insert(0, "model", method)
    history_out["selected_features"] = history_out["selected_features"].apply(" ".join)
    history_out["selected_idx"] = history_out["selected_idx"].apply(
        lambda v: " ".join(str(i) for i in v)
    )

    return ModelRun(
        key=method,
        label=label,
        subset=design.subset,
        estimator=selected,
        config={"k": best["k"], "features": selected_features},
        fold_metrics=fold_metrics,
        summary=summary,
        coefficients=coefficients,
        selection_history=history_out,
        diagnostics={
            "search_seconds": seconds,
            "k_selected": best["k"],
            "features": " ".join(selected_features),
        },
    )


def nested_selection_check(
    method: str,
    design,
    *,
    comparison_splits,
    inner_splits: int = N_SPLITS,
    inner_seed: int = 0,
) -> pd.DataFrame:
    """Misura l'ottimismo introdotto dalla selezione delle variabili non annidata.

    Per ogni partizione di confronto la selezione viene rifatta da capo sulla
    sola parte di addestramento di quella partizione, con una cross-validation
    interna sui suoi motori, e il sottoinsieme risultante viene valutato sulla
    parte di verifica, che non ha partecipato alla scelta. Il punteggio medio
    che ne risulta e' privo della distorsione della selezione, e il suo divario
    dal punteggio riportato in tabella e' la stima di quella distorsione.

    Il risultato e' diagnostico: non entra in graduatoria, perche' i quindici
    fold selezionano sottoinsiemi diversi e non identificano un modello di cui
    leggere le variabili.
    """
    X = design.X_train.to_numpy()
    y = np.asarray(design.y_train)
    groups = np.asarray(design.groups_train)
    features = list(design.features)

    rows = []
    for split in comparison_splits:
        inner = make_splits(groups[split.train], n_splits=inner_splits, seeds=(inner_seed,))
        history = SELECTION_METHODS[method](X[split.train], y[split.train], inner, features)
        best = pick_best(history)
        cols = best["selected_idx"]

        outer = build_grams(X, y, [split])[0]
        rows.append(
            {
                "model": method,
                "seed": split.seed,
                "fold": split.fold,
                "k": best["k"],
                "rmse_inner": best["cv_rmse_mean"],
                "rmse_outer": _fold_rmse(outer, tuple(cols)),
                "selected_features": " ".join(best["selected_features"]),
            }
        )
    return pd.DataFrame(rows)


def coefficient_path(estimator_factory, values, X, y, feature_names, param: str = "alpha") -> pd.DataFrame:
    """Percorso dei coefficienti al variare della penalizzazione.

    I coefficienti sono stimati sull'intera parte di addestramento, come nel
    laboratorio: il percorso descrive il comportamento del modello al variare
    della penalizzazione e non e' una stima di prestazione, quindi non richiede
    partizionamento. La standardizzazione e' adattata una sola volta sulla
    stessa matrice, coerentemente.
    """
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)

    records = []
    for value in values:
        model = estimator_factory(**{param: value}).fit(Z, y)
        coef = np.asarray(model.coef_).ravel()
        for name, c in zip(feature_names, coef):
            records.append({param: value, "feature": name, "coef": float(c)})
    return pd.DataFrame(records)


def comparison_table(runs: list[ModelRun]) -> pd.DataFrame:
    """Tabella di confronto del blocco, ordinata sulla metrica di riferimento.

    La colonna `divario_in_dispersioni` riporta la distanza dalla riga migliore
    in unita' di dispersione fra fold. Due modelli il cui divario e' inferiore
    a una dispersione non sono distinguibili sotto questo protocollo, e la
    colonna rende leggibile questa condizione invece di lasciarla dedurre dal
    confronto fra medie e deviazioni standard.

    La dispersione usata come scala combina quella della riga e quella della
    riga migliore, e non e' quella della sola riga: dividendo ciascun divario
    per la propria dispersione l'ordinamento della colonna non seguirebbe
    quello dell'errore, perche' un modello peggiore ma piu' stabile
    risulterebbe piu' vicino di uno migliore e piu' variabile. La quantita' non
    e' un errore standard e non consente test di significativita': i fold
    condividono le righe di addestramento.
    """
    table = pd.DataFrame([run.summary for run in runs])
    columns = [
        "subset",
        "label",
        "model",
        "config",
        "n_nonzero",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "n_fit",
        "n_configurations",
        "search_seconds",
        "fit_seconds_total",
    ]
    table = table[[c for c in columns if c in table.columns]]
    table = table.sort_values("rmse_mean").reset_index(drop=True)

    best_mean = table["rmse_mean"].iloc[0]
    best_std = table["rmse_std"].iloc[0]
    scale = np.sqrt((table["rmse_std"] ** 2 + best_std**2) / 2.0).replace(0.0, np.nan)
    table.insert(
        table.columns.get_loc("rmse_std") + 1,
        "divario_in_dispersioni",
        ((table["rmse_mean"] - best_mean) / scale).round(2),
    )
    return table


def _count_nonzero(coefficients: pd.DataFrame | None) -> float:
    if coefficients is None:
        return np.nan
    if "zero" not in coefficients:
        return float(len(coefficients))
    return float((~coefficients["zero"]).sum())