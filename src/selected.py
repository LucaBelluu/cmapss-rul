"""Ricostruzione dei modelli selezionati dagli artefatti del confronto.

Ruolo nel progetto
    I quattro blocchi hanno selezionato una configurazione per ciascun modello e
    ne hanno registrato l'etichetta nella colonna `config` delle tabelle. Questo
    modulo ricostruisce da quell'etichetta lo stimatore corrispondente, pronto
    per essere riaddestrato. E' il ponte fra la graduatoria, che e' fatta di
    tabelle, e le letture della fase di chiusura, che hanno bisogno di modelli.

Cosa riceve
    Il nome di un sottoinsieme, la struttura `Design` corrispondente e la
    graduatoria complessiva prodotta da `scripts/run_final_ranking.py`.

Cosa produce
    Pipeline non adattate, identiche a quelle valutate nei blocchi. Non addestra
    e non scrive su disco.

Come avviene la ricostruzione
    Non leggendo i parametri dalle tabelle, ma rigenerando la griglia dal
    registro dei modelli e cercando la combinazione la cui etichetta coincide
    con quella registrata. La scelta ha tre conseguenze volute.

    I valori usati sono quelli del registro a precisione piena, non quelli
    arrotondati a quattro cifre significative dall'etichetta. Ricostruire una
    penalizzazione di 0,08685 dalla stringa darebbe un modello diverso da quello
    valutato.

    La corrispondenza deve essere unica: se nessuna combinazione della griglia
    produce l'etichetta registrata, o se piu' di una la produce, la ricostruzione
    fallisce invece di scegliere. Il primo caso si verifica se il registro
    attualmente nella repository non e' quello che ha prodotto gli artefatti, e
    e' quindi anche un controllo di coerenza fra codice e risultati.

    Il caso di parita' fra configurazioni e' risolto correttamente. Su FD001 due
    combinazioni della griglia della rete hanno lo stesso punteggio fino alla
    quattordicesima cifra, e la riga di rango primo nella tabella della griglia
    non identifica quindi da sola quella valutata: l'etichetta registrata nella
    tabella di confronto sì.

Modelli senza griglia
    La regressione lineare multipla, il bagging di alberi e le due baseline non
    hanno iperparametri: la loro etichetta e' costante e la ricostruzione si
    riduce alla composizione della pipeline.

Metodi di selezione delle variabili
    Non hanno una griglia ma un percorso di ricerca, registrato in
    `{SUBSET}_selection_history.csv`. Il modello selezionato e' la regressione
    lineare sulle colonne del passo di errore minimo, e la cardinalita'
    ricostruita viene confrontata con quella dell'etichetta.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid

from src.baselines import all_baselines
from src.data import PROJECT_ROOT

# La regola con cui una configurazione diventa etichetta deve avere una sola
# definizione: reimplementarla qui produrrebbe due formattazioni che possono
# divergere senza che nulla lo segnali, e la ricostruzione poggia proprio sulla
# coincidenza fra le due.
from src.experiment import _config_label
from src.final import BASELINE_KEYS
from src.pipeline import build_pipeline
from src.registry import (
    KERNEL_MODELS,
    LINEAR_MODELS,
    NONLINEAR_MODELS,
    SELECTION_MODELS,
    TREE_MODELS,
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Cartella da cui provengono gli artefatti dei metodi di selezione.
SELECTION_DIR = "linear_models"

NO_HYPERPARAMETERS = "nessun iperparametro"


def _merged_registry() -> dict:
    """Registro unico dei modelli con griglia, sui quattro blocchi.

    Le chiavi devono essere distinte fra blocchi: due modelli con lo stesso
    identificativo renderebbero ambigua ogni lettura che parte dalla
    graduatoria.
    """
    registry: dict = {}
    for block in (LINEAR_MODELS, NONLINEAR_MODELS, TREE_MODELS, KERNEL_MODELS):
        for key, spec in block.items():
            if key in registry:
                raise AssertionError(f"identificativo {key} presente in piu' di un blocco")
            registry[key] = spec
    return registry


REGISTRY = _merged_registry()


def load_ranking(subset: str) -> pd.DataFrame:
    """Graduatoria complessiva, prodotta da `scripts/run_final_ranking.py`."""
    path = EXPERIMENTS_DIR / "final" / f"{subset}_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"manca {path}: la graduatoria va prodotta prima di ricostruire i modelli"
        )
    return pd.read_csv(path)


def _match_configuration(spec, config: str, n_features: int) -> dict:
    """Combinazione della griglia la cui etichetta coincide con quella registrata."""
    grid = spec.param_grid(n_features)
    if not grid:
        if config != NO_HYPERPARAMETERS:
            raise AssertionError(
                f"{spec.key}: la tabella registra la configurazione '{config}' ma il "
                f"registro non prevede iperparametri per questo modello"
            )
        return {}

    matches = [c for c in ParameterGrid(grid) if _config_label(c) == config]
    if not matches:
        raise AssertionError(
            f"{spec.key}: nessuna configurazione della griglia attuale produce "
            f"l'etichetta '{config}'. Il registro nella repository non e' quello "
            f"che ha prodotto gli artefatti."
        )
    if len(matches) > 1:
        raise AssertionError(
            f"{spec.key}: {len(matches)} configurazioni della griglia producono "
            f"l'etichetta '{config}', che non le distingue"
        )
    return matches[0]


def _selection_features(subset: str, key: str, config: str) -> list[str]:
    """Colonne selezionate dal percorso di ricerca di un metodo di selezione.

    Il criterio e' quello della ricerca: il passo di errore minimo lungo la
    storia. La cardinalita' ottenuta viene confrontata con quella registrata
    nell'etichetta, che e' l'unico controllo disponibile sul fatto che la storia
    su disco sia quella che ha prodotto la riga della tabella.
    """
    path = EXPERIMENTS_DIR / SELECTION_DIR / f"{subset}_selection_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"manca {path}")

    history = pd.read_csv(path)
    history = history[history["model"] == key]
    if history.empty:
        raise AssertionError(f"{subset}: il percorso di ricerca di {key} non e' negli artefatti")

    row = history.loc[history["cv_rmse_mean"].idxmin()]
    features = str(row["selected_features"]).split()
    expected = _config_label({"k": int(row["k"])})
    if config != expected:
        raise AssertionError(
            f"{key} su {subset}: il percorso di ricerca su disco seleziona {expected}, "
            f"la tabella registra {config}"
        )
    if len(features) != int(row["k"]):
        raise AssertionError(
            f"{key} su {subset}: {len(features)} colonne per una cardinalita' di {row['k']}"
        )
    return features


def rebuild(key: str, config: str, design):
    """Stimatore corrispondente alla configurazione selezionata, non adattato.

    Ritorna una pipeline identica a quella valutata nel blocco di provenienza,
    perche' costruita dallo stesso registro e con la stessa composizione di
    pre-processing.
    """
    if key in BASELINE_KEYS:
        return all_baselines()[key]

    if key in SELECTION_MODELS:
        features = _selection_features(design.subset, key, config)
        return build_pipeline(LinearRegression(), columns=features)

    if key not in REGISTRY:
        raise KeyError(f"{key} non e' nel registro dei modelli")

    spec = REGISTRY[key]
    parameters = _match_configuration(spec, config, len(design.features))
    pipeline = build_pipeline(clone(spec.estimator))
    return pipeline.set_params(**parameters) if parameters else pipeline


def rebuild_all(design, ranking: pd.DataFrame | None = None) -> list[dict]:
    """Ricostruisce ogni riga della graduatoria, nell'ordine della graduatoria.

    Ritorna un record per modello con identificativo, etichetta estesa, blocco di
    provenienza, configurazione e stimatore. La ricostruzione dell'intera
    graduatoria e' anche la verifica che il registro nella repository sia
    coerente con tutti gli artefatti prodotti, e non solo con quelli del blocco
    che si sta leggendo.
    """
    ranking = load_ranking(design.subset) if ranking is None else ranking
    records = []
    for _, row in ranking.iterrows():
        records.append(
            {
                "model": row["model"],
                "label": row["label"],
                "blocco": row.get("blocco", ""),
                "config": row["config"],
                "rmse_cv_mean": row["rmse_mean"],
                "rmse_cv_std": row["rmse_std"],
                "estimator": rebuild(row["model"], row["config"], design),
            }
        )
    return records
