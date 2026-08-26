"""Verifica di equivalenza del motore di stima usato dalla selezione delle variabili.

Ruolo nel progetto
    `src.selection` non stima i minimi quadrati costruendo una pipeline per
    ogni sottoinsieme, ma risolvendo il sistema normale sulle sottomatrici
    delle statistiche sufficienti di ciascuna partizione. La riformulazione e'
    algebricamente esatta e riduce di due ordini di grandezza il costo della
    ricerca esaustiva, ma e' codice del progetto e non una funzione di
    libreria: senza una verifica, un errore nella riformulazione produrrebbe
    numeri plausibili e sbagliati.

Cosa verifica
    1. Che l'errore calcolato dal motore su un sottoinsieme di variabili
       coincida, entro la tolleranza dell'aritmetica in virgola mobile, con
       quello prodotto dalla valutazione ordinaria della corrispondente
       pipeline sotto `src.protocol.evaluate`. E' il controllo che lega il
       motore al protocollo con cui sono valutati tutti gli altri modelli.
    2. Che la ricerca esaustiva sul motore restituisca gli stessi sottoinsiemi
       della ricerca esaustiva ingenua, su un numero di variabili abbastanza
       piccolo da rendere eseguibili entrambe.
    3. Che i tre metodi di selezione producano percorsi coerenti fra loro:
       forward e backward non peggiorano il modello completo, e nessuno dei
       due batte la ricerca esaustiva a parita' di cardinalita', che e' una
       proprieta' vera per costruzione e falsificabile dal codice.

Come si lancia
    python -m scripts.run_selection_check
    python -m scripts.run_selection_check --synthetic

    Senza argomenti la verifica avviene sulla matrice di progetto di FD001,
    che richiede i dati grezzi. Con `--synthetic` avviene su dati generati, che
    non li richiedono: le due varianti verificano la stessa proprieta'
    algebrica.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression

from src.data import CYCLE_COL
from src.pipeline import build_pipeline
from src.protocol import evaluate, make_splits
from src.selection import (
    backward_stepwise,
    best_subset,
    build_grams,
    cv_rmse,
    forward_stepwise,
)

TOLERANCE = 1e-8


def synthetic_design(n_units: int = 40, n_features: int = 8, seed: int = 0):
    """Matrice raggruppata per unita', con struttura simile a quella del progetto.

    Serve unicamente alla verifica: le righe di una stessa unita' sono
    correlate fra loro, come i cicli di uno stesso motore, cosi' che il
    controllo avvenga su una matrice con lo stesso tipo di condizionamento.
    """
    rng = np.random.default_rng(seed)
    n_sensors = n_features - 1
    frames, targets, groups = [], [], []
    for unit in range(n_units):
        length = int(rng.integers(60, 140))
        drift = np.linspace(0.0, 1.0, length)[:, None]
        base = rng.normal(size=(1, n_sensors))
        noise = rng.normal(scale=0.3, size=(length, n_sensors))
        sensors = base + drift * rng.normal(size=(1, n_sensors)) + noise
        cycles = np.arange(1, length + 1)[:, None]
        frames.append(np.hstack([cycles, sensors]))
        targets.append(np.maximum(length - np.arange(length), 0) + rng.normal(scale=2.0, size=length))
        groups.append(np.full(length, unit))
    # La prima colonna e' il numero di ciclo, come nella matrice di progetto:
    # senza di essa la baseline che usa il solo numero di ciclo non e'
    # costruibile e la catena non sarebbe esercitata per intero.
    columns = [CYCLE_COL] + [f"x_{i:02d}" for i in range(n_sensors)]
    X = pd.DataFrame(np.vstack(frames), columns=columns)
    return X, np.concatenate(targets), np.concatenate(groups)


def check_against_protocol(X, y, groups, splits, feature_names, n_checks: int = 12) -> None:
    """Confronta il motore con la valutazione ordinaria su sottoinsiemi casuali."""
    grams = build_grams(X, y, splits)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_checks):
        k = int(rng.integers(1, len(feature_names) + 1))
        cols = sorted(rng.choice(len(feature_names), size=k, replace=False).tolist())
        names = [feature_names[i] for i in cols]

        fast, _ = cv_rmse(grams, cols)
        reference = evaluate(
            build_pipeline(LinearRegression(), columns=names), X, y, groups, splits
        )["rmse"].mean()
        rows.append({"k": k, "motore": fast, "protocollo": reference, "scarto": abs(fast - reference)})

    table = pd.DataFrame(rows)
    print("\nequivalenza con la valutazione ordinaria")
    print(table.to_string(index=False))
    worst = table["scarto"].max()
    if worst > TOLERANCE:
        raise AssertionError(f"scarto massimo {worst:.3e} oltre la tolleranza {TOLERANCE:.0e}")
    print(f"scarto massimo {worst:.3e}, entro la tolleranza")


def naive_best_subset(X, y, groups, splits, feature_names) -> pd.DataFrame:
    """Ricerca esaustiva nella forma del laboratorio, usata come riferimento."""
    records = []
    for k in range(1, len(feature_names) + 1):
        best_names, best_score = None, np.inf
        for cols in combinations(range(len(feature_names)), k):
            names = [feature_names[i] for i in cols]
            score = evaluate(
                build_pipeline(LinearRegression(), columns=names), X, y, groups, splits
            )["rmse"].mean()
            if score < best_score:
                best_score, best_names = score, names
        records.append({"k": k, "selected_features": best_names, "cv_rmse_mean": best_score})
    return pd.DataFrame(records)


def check_exhaustive(X, y, groups, splits, feature_names) -> None:
    """Confronta ricerca esaustiva veloce e ingenua sullo stesso spazio."""
    fast = best_subset(X, y, splits, feature_names)
    slow = naive_best_subset(X, y, groups, splits, feature_names)

    merged = fast.merge(slow, on="k", suffixes=("_motore", "_ingenua"))
    merged["stesso_sottoinsieme"] = [
        sorted(a) == sorted(b)
        for a, b in zip(merged["selected_features_motore"], merged["selected_features_ingenua"])
    ]
    merged["scarto"] = (merged["cv_rmse_mean_motore"] - merged["cv_rmse_mean_ingenua"]).abs()
    print("\nequivalenza della ricerca esaustiva")
    print(
        merged[["k", "cv_rmse_mean_motore", "cv_rmse_mean_ingenua", "scarto", "stesso_sottoinsieme"]]
        .to_string(index=False)
    )
    if not merged["stesso_sottoinsieme"].all():
        raise AssertionError("le due ricerche selezionano sottoinsiemi diversi")
    if merged["scarto"].max() > TOLERANCE:
        raise AssertionError(f"scarto massimo {merged['scarto'].max():.3e} oltre la tolleranza")
    print("stessi sottoinsiemi a ogni cardinalita'")


def check_paths(X, y, splits, feature_names) -> None:
    """Controlla la coerenza reciproca dei tre percorsi di ricerca."""
    exhaustive = best_subset(X, y, splits, feature_names).set_index("k")["cv_rmse_mean"]
    forward = forward_stepwise(X, y, splits, feature_names).set_index("k")["cv_rmse_mean"]
    backward = backward_stepwise(X, y, splits, feature_names).set_index("k")["cv_rmse_mean"]

    table = pd.DataFrame(
        {"esaustiva": exhaustive, "forward": forward, "backward": backward}
    ).sort_index()
    table["forward_non_batte"] = table["forward"] >= table["esaustiva"] - TOLERANCE
    table["backward_non_batte"] = table["backward"] >= table["esaustiva"] - TOLERANCE
    print("\ncoerenza dei tre percorsi (rmse medio per cardinalita')")
    print(table.to_string())

    if not table["forward_non_batte"].all() or not table["backward_non_batte"].all():
        raise AssertionError(
            "una ricerca greedy batte la ricerca esaustiva: impossibile per costruzione"
        )
    print("nessuna ricerca greedy batte la ricerca esaustiva")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="verifica su dati generati")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument(
        "--pool",
        type=int,
        default=8,
        help="numero di variabili su cui eseguire anche la ricerca ingenua",
    )
    args = parser.parse_args()

    if args.synthetic:
        X, y, groups = synthetic_design()
    else:
        from src.design import build_design

        design = build_design(args.subset)
        X, y, groups = design.X_train, design.y_train, design.groups_train

    splits = make_splits(groups, seeds=(0,))
    feature_names = list(X.columns)
    print(f"matrice: {X.shape[0]} righe, {len(feature_names)} variabili, {len(splits)} partizioni")

    check_against_protocol(X, y, groups, splits, feature_names)

    pool = feature_names[: args.pool]
    print(f"\nricerca esaustiva ingenua limitata a {len(pool)} variabili: {2 ** len(pool) - 1} sottoinsiemi")
    check_exhaustive(X[pool], y, groups, splits, pool)
    check_paths(X[pool], y, splits, pool)

    print("\ntutti i controlli superati")


if __name__ == "__main__":
    main()