"""Metodi di ricampionamento per la stima dell'errore, applicati alle unita' motore.

Ruolo nel progetto
    Riproduce i quattro metodi di stima dell'errore del laboratorio 6
    (partizione unica addestramento e verifica, esclusione di una unita' per
    volta, K-Fold, bootstrap) su un unico modello, la regressione lineare
    multipla. Il confronto non riguarda i modelli ma le procedure con cui il
    loro errore viene stimato: mostra quanto la stima dipenda dalla procedura
    scelta, ed e' la giustificazione empirica dello schema adottato dal
    protocollo del progetto.

Cosa riceve
    La matrice di progetto di un sottoinsieme e uno stimatore. Ogni funzione
    costruisce le proprie partizioni e le valuta attraverso
    `src.protocol.evaluate`, quindi metriche e clonazione dello stimatore sono
    le stesse usate da tutti gli esperimenti.

Cosa produce
    Per ciascun metodo, le metriche di ogni ripetizione e un riepilogo su media
    e dispersione. Per il bootstrap anche la distribuzione dei coefficienti.

Trasposizione alle unita'
    Nel laboratorio i quattro metodi ricampionano righe, perche' le
    osservazioni sono indipendenti. Qui l'unita' di ricampionamento e' il
    motore, per la stessa ragione per cui il partizionamento del protocollo
    avviene per motore: le righe di una stessa traiettoria sono cicli
    consecutivi e non sono indipendenti fra loro. L'esclusione di una
    osservazione per volta diventa percio' esclusione di un motore per volta,
    che e' la trasposizione della validazione incrociata esaustiva a dati
    raggruppati.

    La trasposizione ha una conseguenza sulla numerosita': i metodi operano su
    100 unita' e non su 20.631 righe, e le stime che ne derivano hanno la
    variabilita' che compete a un campione di cento elementi.

Il bootstrap
    Nel laboratorio il bootstrap e' una funzione di ricampionamento scritta da
    zero, che riceve un insieme di dati e restituisce un numero richiesto di
    campioni estratti con reinserimento. La funzione `bootstrap` di questo
    modulo conserva quella firma e viene applicata all'elenco degli
    identificativi dei motori.

    Serve a due scopi distinti. Il primo e' la variabilita' dei coefficienti
    della regressione: rieseguendo la stima su ogni campione si ottiene la
    distribuzione di ciascun coefficiente, da cui si legge quali variabili
    hanno un contributo di segno stabile e quali cambiano segno al variare del
    campione. Il secondo e' la stima dell'errore fuori campione: i motori non
    estratti in un campione non hanno partecipato all'addestramento e formano
    una parte di verifica, il che rende il bootstrap confrontabile con gli
    altri tre metodi nella stessa tabella.

    Un campione bootstrap estratto con reinserimento da n unita' ne lascia
    fuori in media una frazione pari a circa il 37 per cento, che e' la
    dimensione attesa della parte di verifica di ciascuna ripetizione.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from src.protocol import FoldSplit, evaluate, make_splits, regression_metrics, summarize

# Numero di ripetizioni della partizione unica. Il laboratorio ne usa venti per
# rendere visibile la dipendenza della stima dalla particolare partizione.
N_VALIDATION_REPEATS = 20
VALIDATION_TEST_SHARE = 0.3

# Numero di campioni bootstrap. Il valore e' un compromesso fra stabilita'
# della distribuzione dei coefficienti e costo: duecento riaddestramenti di una
# regressione lineare su ventimila righe restano nell'ordine dei secondi.
N_BOOTSTRAP = 200


def validation_set_approach(
    estimator,
    X,
    y,
    groups,
    *,
    n_repeats: int = N_VALIDATION_REPEATS,
    test_share: float = VALIDATION_TEST_SHARE,
) -> pd.DataFrame:
    """Partizione unica in addestramento e verifica, ripetuta su piu' semi.

    La partizione separa i motori e non le righe. La ripetizione su semi
    diversi non serve a ottenere una stima migliore ma a rendere misurabile
    quanto la stima dipenda dalla partizione scelta, che e' il limite del
    metodo.
    """
    groups = np.asarray(groups)
    splits = []
    for seed in range(n_repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
        train_idx, valid_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
        splits.append(FoldSplit(seed=seed, fold=0, train=train_idx, valid=valid_idx))

    metrics = evaluate(estimator, X, y, groups, splits)
    metrics.insert(0, "method", "validation_set")
    return metrics


def leave_one_unit_out(estimator, X, y, groups) -> pd.DataFrame:
    """Esclusione di un motore per volta: tante stime quante sono le unita'.

    E' la trasposizione ai dati raggruppati della validazione incrociata
    esaustiva del laboratorio. Ogni parte di verifica e' una traiettoria
    intera, quindi le metriche per fold sono calcolate su alcune centinaia di
    righe e non su una sola osservazione.
    """
    groups = np.asarray(groups)
    splitter = LeaveOneGroupOut()
    splits = [
        FoldSplit(seed=0, fold=fold, train=train_idx, valid=valid_idx)
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(np.zeros(len(groups)), groups=groups)
        )
    ]
    metrics = evaluate(estimator, X, y, groups, splits)
    metrics.insert(0, "method", "leave_one_unit_out")
    return metrics


def k_fold(estimator, X, y, groups, *, n_splits: int, seeds=(0, 1, 2)) -> pd.DataFrame:
    """K-Fold con vincolo di gruppo, al numero di fold indicato."""
    splits = make_splits(groups, n_splits=n_splits, seeds=seeds)
    metrics = evaluate(estimator, X, y, groups, splits)
    metrics.insert(0, "method", f"kfold_{n_splits}")
    return metrics


def bootstrap(data, n_samples: int, random_state: int = 0) -> list[list]:
    """Ricampionamento con reinserimento: restituisce `n_samples` campioni.

    Riproduce la funzione richiesta dal laboratorio: riceve un insieme di dati
    in forma di elenco, il numero di campioni desiderato e un seme, e
    restituisce l'elenco dei campioni. Ogni campione ha la stessa numerosita'
    dell'insieme di partenza ed e' estratto con reinserimento, quindi puo'
    contenere ripetizioni e lasciare fuori parte degli elementi.
    """
    rng = np.random.default_rng(random_state)
    items = list(data)
    n = len(items)
    return [[items[i] for i in rng.integers(0, n, size=n)] for _ in range(n_samples)]


def bootstrap_estimates(
    estimator,
    X,
    y,
    groups,
    feature_names,
    *,
    n_samples: int = N_BOOTSTRAP,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applica il bootstrap ai motori, per i coefficienti e per l'errore.

    Per ciascun campione le righe dei motori estratti formano la parte di
    addestramento, con le ripetizioni che il reinserimento comporta, e le
    righe dei motori mai estratti la parte di verifica. Un campione che non
    lascia fuori alcun motore non produce una stima dell'errore e viene
    escluso dal solo conteggio dell'errore, non da quello dei coefficienti.

    Restituisce la distribuzione dei coefficienti e le metriche per campione.
    """
    groups = np.asarray(groups)
    units = np.unique(groups)
    rows_by_unit = {unit: np.flatnonzero(groups == unit) for unit in units}

    coefficient_rows = []
    metric_rows = []
    for b, sample in enumerate(bootstrap(units, n_samples, random_state)):
        train_idx = np.concatenate([rows_by_unit[u] for u in sample])
        left_out = [u for u in units if u not in set(sample)]

        model = clone(estimator)
        model.fit(X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx], np.asarray(y)[train_idx])

        coef = np.asarray(model.named_steps["model"].coef_).ravel()
        for name, value in zip(feature_names, coef):
            coefficient_rows.append({"sample": b, "feature": name, "coef": float(value)})

        if not left_out:
            continue
        valid_idx = np.concatenate([rows_by_unit[u] for u in left_out])
        y_pred = model.predict(X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx])
        record = {
            "method": "bootstrap",
            "sample": b,
            "n_train_rows": len(train_idx),
            "n_valid_rows": len(valid_idx),
            "n_train_units": len(set(sample)),
            "n_valid_units": len(left_out),
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
        }
        record.update(regression_metrics(np.asarray(y)[valid_idx], y_pred))
        metric_rows.append(record)

    return pd.DataFrame(coefficient_rows), pd.DataFrame(metric_rows)


def coefficient_intervals(coefficients: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Riepilogo della distribuzione bootstrap di ciascun coefficiente.

    L'intervallo e' costruito sui quantili empirici della distribuzione. La
    colonna `stable_sign` indica se l'intervallo esclude lo zero, cioe' se il
    contributo della variabile mantiene lo stesso segno al variare del
    campione di motori. Non e' un test di ipotesi: e' una lettura della
    variabilita' della stima, e come tale entra nel commento del modello.
    """
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    summary = (
        coefficients.groupby("feature")["coef"]
        .agg(
            mean="mean",
            std="std",
            lower=lambda s: s.quantile(lower_q),
            upper=lambda s: s.quantile(upper_q),
        )
        .reset_index()
    )
    summary["stable_sign"] = (summary["lower"] > 0) | (summary["upper"] < 0)
    summary["abs_mean"] = summary["mean"].abs()
    return summary.sort_values("abs_mean", ascending=False).reset_index(drop=True)


def summarize_methods(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Riepiloga i metodi in una tabella unica, un metodo per riga.

    La dispersione riportata da ciascun metodo non ha lo stesso significato:
    per la partizione unica descrive la variabilita' fra partizioni diverse,
    per il K-Fold e per l'esclusione di una unita' per volta la variabilita'
    fra parti di verifica di una stessa procedura, per il bootstrap la
    variabilita' fra campioni. I valori sono percio' accostabili ma non
    intercambiabili, e la tabella li tiene distinti nella colonna `n_fit`, che
    dice su quante stime ciascuna riga e' costruita.
    """
    rows = []
    for frame in frames:
        method = frame["method"].iloc[0]
        summary = summarize(frame.drop(columns=["method"]), label=method)
        summary["n_valid_rows_mean"] = frame["n_valid_rows"].mean() if "n_valid_rows" in frame else np.nan
        rows.append(summary)
    table = pd.DataFrame(rows).rename(columns={"model": "method"})
    return table