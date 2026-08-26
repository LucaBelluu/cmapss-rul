"""Selezione delle variabili: best subset, forward stepwise, backward stepwise.

Ruolo nel progetto
    Implementa i tre metodi di selezione del sottoinsieme di variabili del
    laboratorio 7, sotto il protocollo di valutazione del progetto. I tre
    metodi condividono lo stesso motore di stima e differiscono unicamente per
    la strategia con cui esplorano lo spazio dei sottoinsiemi: la differenza
    fra i loro risultati e' quindi attribuibile alla ricerca e non al modo in
    cui il modello viene stimato.

Cosa riceve
    La matrice di progetto, il target, e le partizioni prodotte da
    `src.protocol.make_splits`. La selezione avviene sulle partizioni del seme
    dedicato alla ricerca, come per gli iperparametri di ogni altro modello.

Cosa produce
    Per ciascun metodo, la storia della ricerca in forma tabellare (un record
    per cardinalita' esplorata, con il migliore sottoinsieme trovato e il suo
    errore in cross-validation) e il sottoinsieme selezionato, cioe' quello
    con l'errore minimo lungo la storia.

Perche' un motore di stima dedicato
    La ricerca esaustiva su p variabili richiede 2^p - 1 stime per ogni
    partizione: 262.143 su FD001 e 524.287 su FD003, moltiplicate per il
    numero di fold. Nella forma del laboratorio, dove ogni sottoinsieme viene
    valutato costruendo una pipeline e chiamando la cross-validation, il costo
    non e' sostenibile.

    Il costo si abbatte osservando che i minimi quadrati su un sottoinsieme di
    variabili si ottengono dalle sottomatrici di X'X e X'y, che dipendono dalla
    partizione e non dal sottoinsieme, e si calcolano quindi una volta sola per
    fold. Lo stesso vale per l'errore sulla parte di verifica del fold, che si
    scrive come forma quadratica nei coefficienti e non richiede di calcolare
    le predizioni riga per riga. Il costo per sottoinsieme passa dall'ordine
    del numero di righe all'ordine del quadrato del numero di variabili
    selezionate, ed e' cio' che rende eseguibile la ricerca esaustiva completa
    invece di una sua versione troncata a una cardinalita' massima arbitraria.

    La riformulazione e' algebricamente esatta e non e' un'approssimazione. La
    verifica di equivalenza contro la stima ordinaria e' in
    `scripts/run_selection_check.py`.

Standardizzazione
    Il motore centra e riduce le variabili con media e deviazione standard
    calcolate sulla sola parte di addestramento della partizione, e centra il
    target sulla stessa parte. Riproduce quindi esattamente la composizione di
    standardizzazione e regressione lineare usata da tutti gli altri modelli
    del confronto, con il pre-processing dentro il flusso di validazione.
    Il centraggio del target sostituisce il termine di intercetta, che non
    compare percio' fra le incognite del sistema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

# Tolleranza sotto la quale la deviazione standard di una colonna e'
# considerata nulla. Le colonne costanti sono gia' rimosse dalla matrice di
# progetto, quindi il caso non si presenta: la soglia evita una divisione per
# zero se il motore viene riusato su matrici costruite altrove.
_STD_FLOOR = 1e-12


@dataclass(frozen=True)
class FoldGram:
    """Statistiche sufficienti di una partizione, calcolate una volta sola.

    gram, moment
        Prodotti X'X e X'y sulla parte di addestramento, standardizzata.
    gram_valid, moment_valid, tss_valid
        Le quantita' corrispondenti sulla parte di verifica, con il target
        centrato sulla media di addestramento, e la somma dei suoi quadrati.
        Permettono di calcolare la somma dei quadrati dei residui di un
        qualunque sottoinsieme senza costruire le predizioni.
    n_valid
        Numero di righe di verifica, necessario per passare dalla somma dei
        quadrati dei residui alla radice dell'errore quadratico medio.
    """

    gram: np.ndarray = field(repr=False)
    moment: np.ndarray = field(repr=False)
    gram_valid: np.ndarray = field(repr=False)
    moment_valid: np.ndarray = field(repr=False)
    tss_valid: float
    n_valid: int


def build_grams(X, y, splits) -> list[FoldGram]:
    """Precalcola le statistiche sufficienti di ogni partizione.

    E' l'unico punto in cui si scorrono le righe: da qui in avanti il costo di
    valutare un sottoinsieme non dipende piu' dal numero di righe.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    grams: list[FoldGram] = []
    for s in splits:
        X_tr, y_tr = X[s.train], y[s.train]
        X_va, y_va = X[s.valid], y[s.valid]

        center = X_tr.mean(axis=0)
        scale = X_tr.std(axis=0, ddof=0)
        scale = np.where(scale < _STD_FLOOR, 1.0, scale)
        y_center = y_tr.mean()

        Z_tr = (X_tr - center) / scale
        Z_va = (X_va - center) / scale
        r_tr = y_tr - y_center
        r_va = y_va - y_center

        grams.append(
            FoldGram(
                gram=Z_tr.T @ Z_tr,
                moment=Z_tr.T @ r_tr,
                gram_valid=Z_va.T @ Z_va,
                moment_valid=Z_va.T @ r_va,
                tss_valid=float(r_va @ r_va),
                n_valid=len(s.valid),
            )
        )
    return grams


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Coefficienti dei minimi quadrati per un sottoinsieme di variabili.

    Il ripiego sui minimi quadrati in forma generale copre il caso di
    collinearita' esatta, in cui il sistema e' singolare: restituisce la
    soluzione di norma minima, come fa la regressione lineare di scikit-learn.
    Sulle matrici del progetto il caso non si presenta, la coppia di sensori
    piu' correlata essendo a 0,963.
    """
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _fold_rmse(gram: FoldGram, cols: tuple[int, ...] | list[int]) -> float:
    """Errore quadratico medio di un sottoinsieme su una partizione.

    La somma dei quadrati dei residui sulla parte di verifica si ottiene dalla
    forma quadratica dei coefficienti, senza costruire le predizioni. Il valore
    puo' risultare negativo per soli errori di arrotondamento quando l'errore
    e' prossimo a zero, ed e' percio' troncato a zero prima della radice.
    """
    idx = np.ix_(cols, cols)
    beta = _solve(gram.gram[idx], gram.moment[list(cols)])
    rss = (
        gram.tss_valid
        - 2.0 * float(beta @ gram.moment_valid[list(cols)])
        + float(beta @ (gram.gram_valid[idx] @ beta))
    )
    return float(np.sqrt(max(rss, 0.0) / gram.n_valid))


def cv_rmse(grams: list[FoldGram], cols) -> tuple[float, float]:
    """Media e deviazione standard dell'errore di un sottoinsieme sulle partizioni.

    La media sui fold e' la quantita' su cui i tre metodi confrontano i
    sottoinsiemi, coerentemente con il resto del progetto, dove le metriche
    sono calcolate per fold e poi mediate.
    """
    scores = [_fold_rmse(g, tuple(cols)) for g in grams]
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


def best_subset(X, y, splits, feature_names) -> pd.DataFrame:
    """Ricerca esaustiva su tutti i sottoinsiemi non vuoti di variabili.

    Restituisce un record per cardinalita', con il sottoinsieme di errore
    minimo a quella cardinalita'. La sequenza dei minimi per cardinalita' e' il
    materiale con cui si legge il compromesso fra numero di variabili e errore,
    ed e' la stessa forma prodotta dal laboratorio.

    Il numero di sottoinsiemi cresce come 2^p: la funzione e' eseguibile sulle
    18 e 19 variabili del progetto e non lo sarebbe su un numero
    sensibilmente maggiore.
    """
    grams = build_grams(X, y, splits)
    n_features = len(feature_names)

    records = []
    for k in range(1, n_features + 1):
        best_cols: tuple[int, ...] | None = None
        best_score = np.inf
        for cols in combinations(range(n_features), k):
            score = float(np.mean([_fold_rmse(g, cols) for g in grams]))
            if score < best_score:
                best_score = score
                best_cols = cols
        mean, std = cv_rmse(grams, best_cols)
        records.append(
            {
                "k": k,
                "n_subsets": _binomial(n_features, k),
                "selected_idx": list(best_cols),
                "selected_features": [feature_names[i] for i in best_cols],
                "cv_rmse_mean": mean,
                "cv_rmse_std": std,
            }
        )
    return pd.DataFrame(records)


def forward_stepwise(X, y, splits, feature_names) -> pd.DataFrame:
    """Aggiunge a ogni passo la variabile che riduce di piu' l'errore.

    Il percorso viene costruito per intero, fino al modello completo, senza
    arresto anticipato al primo passo che non migliora. Il costo e' lo stesso a
    meno di poche stime, e il percorso completo rende leggibile l'andamento
    dell'errore oltre il minimo, che e' materiale di commento. La selezione
    avviene poi sul passo di errore minimo, come nel laboratorio.
    """
    grams = build_grams(X, y, splits)
    n_features = len(feature_names)

    selected: list[int] = []
    remaining = list(range(n_features))
    records = []
    while remaining:
        best_j, best_score = None, np.inf
        for j in remaining:
            score = float(np.mean([_fold_rmse(g, tuple(selected + [j])) for g in grams]))
            if score < best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)
        mean, std = cv_rmse(grams, selected)
        records.append(
            {
                "step": len(selected),
                "k": len(selected),
                "changed_feature": feature_names[best_j],
                "selected_idx": selected.copy(),
                "selected_features": [feature_names[i] for i in selected],
                "cv_rmse_mean": mean,
                "cv_rmse_std": std,
            }
        )
    return pd.DataFrame(records)


def backward_stepwise(X, y, splits, feature_names) -> pd.DataFrame:
    """Rimuove a ogni passo la variabile la cui esclusione riduce di piu' l'errore.

    Parte dal modello completo e scende fino al modello a una variabile. Come
    per la forward, il percorso e' costruito per intero e la selezione avviene
    sul passo di errore minimo, cosi' che i due metodi differiscano soltanto
    per la direzione della ricerca e siano confrontabili fra loro.

    Nel materiale del corso la backward e' proposta come esercizio e non
    svolta: l'implementazione e' interamente del progetto.
    """
    grams = build_grams(X, y, splits)
    n_features = len(feature_names)

    selected = list(range(n_features))
    mean, std = cv_rmse(grams, selected)
    records = [
        {
            "step": 0,
            "k": len(selected),
            "changed_feature": "",
            "selected_idx": selected.copy(),
            "selected_features": list(feature_names),
            "cv_rmse_mean": mean,
            "cv_rmse_std": std,
        }
    ]

    step = 0
    while len(selected) > 1:
        step += 1
        best_j, best_score = None, np.inf
        for j in selected:
            candidate = [c for c in selected if c != j]
            score = float(np.mean([_fold_rmse(g, tuple(candidate)) for g in grams]))
            if score < best_score:
                best_score, best_j = score, j
        selected = [c for c in selected if c != best_j]
        mean, std = cv_rmse(grams, selected)
        records.append(
            {
                "step": step,
                "k": len(selected),
                "changed_feature": feature_names[best_j],
                "selected_idx": selected.copy(),
                "selected_features": [feature_names[i] for i in selected],
                "cv_rmse_mean": mean,
                "cv_rmse_std": std,
            }
        )
    return pd.DataFrame(records)


def pick_best(history: pd.DataFrame) -> dict:
    """Estrae dalla storia di una ricerca il sottoinsieme di errore minimo."""
    row = history.loc[history["cv_rmse_mean"].idxmin()]
    return {
        "k": int(row["k"]),
        "selected_idx": list(row["selected_idx"]),
        "selected_features": list(row["selected_features"]),
        "cv_rmse_mean": float(row["cv_rmse_mean"]),
        "cv_rmse_std": float(row["cv_rmse_std"]),
    }


def _binomial(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


SELECTION_METHODS = {
    "best_subset": best_subset,
    "forward_stepwise": forward_stepwise,
    "backward_stepwise": backward_stepwise,
}