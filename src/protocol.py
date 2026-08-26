"""Protocollo di valutazione dei modelli.

Ruolo nel progetto
    Definisce le condizioni sotto cui ogni modello del confronto viene
    valutato. E' l'unico punto in cui sono scritti lo schema di
    partizionamento, il numero di fold, i semi e le metriche: ogni esperimento
    passa da qui, e questo garantisce che il confronto avvenga a parita' di
    condizioni.

Cosa riceve
    Una matrice di progetto X (DataFrame), un target y (array), un vettore
    groups con l'identificativo del motore di ciascuna riga, e uno stimatore
    conforme all'interfaccia di scikit-learn (fit, predict, get_params).

Cosa produce
    Un DataFrame con una riga per fold contenente le metriche, i conteggi di
    righe e di motori e i tempi; su richiesta, le predizioni fuori fold. Non
    scrive su disco: la persistenza e' compito degli script di orchestrazione.

Partizionamento
    Avviene per unita' motore. Le righe di uno stesso motore sono cicli
    consecutivi della stessa traiettoria di degrado e non sono indipendenti:
    una partizione per riga collocherebbe osservazioni quasi identiche su
    entrambi i lati della verifica e produrrebbe una stima sistematicamente
    ottimistica. Lo schema usato (K-Fold con vincolo di gruppo) non compare nei
    laboratori del corso: e' la trasposizione diretta del K-Fold a dati
    raggruppati, resa obbligatoria dalla struttura del dataset.

Selezione e stima
    La cross-validation non e' annidata: le stesse partizioni servono a
    scegliere gli iperparametri e a riportare il punteggio della configurazione
    scelta. Il punteggio riportato e' quindi ottimisticamente distorto. La
    stima non distorta proviene dall'insieme di verifica ufficiale, che non
    entra in nessuna scelta e viene letto una sola volta a graduatoria chiusa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

# Parametri del protocollo. Sono definiti qui e in nessun altro punto del
# progetto: cambiarli qui cambia tutti gli esperimenti in modo coerente.
N_SPLITS = 5
SEARCH_SEEDS = (0,)
COMPARISON_SEEDS = (0, 1, 2)
METRICS = ("rmse", "mae", "r2")


@dataclass(frozen=True)
class FoldSplit:
    """Una singola partizione, con l'indicazione del seme che l'ha generata.

    Tenere seme e indice del fold accanto agli indici permette di riportare la
    dispersione dei risultati distinguendo la variabilita' fra fold dalla
    variabilita' fra ripetizioni della partizione.
    """

    seed: int
    fold: int
    train: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)


def make_splits(groups, n_splits: int = N_SPLITS, seeds=COMPARISON_SEEDS) -> list[FoldSplit]:
    """Costruisce le partizioni per unita' motore, ripetute su piu' semi.

    Ogni motore compare esattamente una volta nella parte di verifica di ogni
    ripetizione, e tutte le sue righe restano dalla stessa parte.
    """
    groups = np.asarray(groups)
    n_units = len(np.unique(groups))
    if n_units < n_splits:
        raise ValueError(f"{n_units} motori non bastano per {n_splits} fold")

    splits: list[FoldSplit] = []
    placeholder = np.zeros(len(groups))
    for seed in seeds:
        # shuffle e random_state sono disponibili su GroupKFold dalla versione
        # 1.6 di scikit-learn e sono cio' che rende ripetibile la partizione su
        # semi diversi.
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(placeholder, groups=groups)):
            splits.append(FoldSplit(seed=seed, fold=fold, train=train_idx, valid=valid_idx))
    return splits


def as_cv(splits) -> list[tuple[np.ndarray, np.ndarray]]:
    """Riduce le partizioni alla forma accettata dal parametro cv di scikit-learn."""
    return [(s.train, s.valid) for s in splits]


def check_no_group_leakage(groups, splits) -> None:
    """Verifica che nessun motore compaia da entrambe le parti di una partizione.

    E' il controllo che rende falsificabile il vincolo su cui poggia l'intera
    valutazione, invece di lasciarlo affidato alla correttezza dello splitter.
    """
    groups = np.asarray(groups)
    for s in splits:
        shared = np.intersect1d(groups[s.train], groups[s.valid])
        if shared.size:
            raise AssertionError(
                f"seme {s.seed} fold {s.fold}: {shared.size} motori presenti "
                f"in addestramento e in verifica"
            )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metriche di regressione del corso.

    RMSE e' la metrica di riferimento: e' nelle unita' del target (cicli) ed e'
    coerente con la perdita minimizzata dalla maggior parte dei modelli in
    confronto. MAE e R quadro accompagnano la lettura e non vengono usate per
    selezionare: il rapporto fra RMSE e MAE dice se l'errore e' dominato da una
    coda di errori grandi, R quadro rende confrontabili insiemi con varianza del
    target diversa.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _rows(data, idx):
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def evaluate(
    estimator,
    X,
    y,
    groups,
    splits,
    *,
    return_predictions: bool = False,
):
    """Valuta uno stimatore su un insieme di partizioni, fold per fold.

    Lo stimatore viene clonato prima di ogni addestramento: l'oggetto passato
    non viene mai adattato, quindi nessuno stato di un fold sopravvive al
    successivo. Le metriche sono calcolate su ciascun fold e restituite
    separatamente, non aggregando le predizioni in un unico vettore: la
    dispersione fra fold e' parte del risultato e va riportata.

    Ritorna il DataFrame delle metriche per fold e, se richiesto, il DataFrame
    delle predizioni fuori fold.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y e groups hanno lunghezze diverse")

    records = []
    predictions = []
    for s in splits:
        model = clone(estimator)

        start = time.perf_counter()
        model.fit(_rows(X, s.train), y[s.train])
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(_rows(X, s.valid))
        predict_seconds = time.perf_counter() - start

        record = {"seed": s.seed, "fold": s.fold}
        record.update(regression_metrics(y[s.valid], y_pred))
        record.update(
            {
                "n_train_rows": len(s.train),
                "n_valid_rows": len(s.valid),
                "n_train_units": len(np.unique(groups[s.train])),
                "n_valid_units": len(np.unique(groups[s.valid])),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        records.append(record)

        if return_predictions:
            predictions.append(
                pd.DataFrame(
                    {
                        "seed": s.seed,
                        "fold": s.fold,
                        "row": s.valid,
                        "unit": groups[s.valid],
                        "y_true": y[s.valid],
                        "y_pred": y_pred,
                    }
                )
            )

    fold_metrics = pd.DataFrame.from_records(records)
    if return_predictions:
        return fold_metrics, pd.concat(predictions, ignore_index=True)
    return fold_metrics


def summarize(fold_metrics: pd.DataFrame, label: str | None = None) -> pd.Series:
    """Riassume le metriche per fold in media e deviazione standard.

    La deviazione standard e' calcolata sui fold e non e' l'errore standard
    della media: i fold condividono le righe di addestramento e non sono
    indipendenti. E' una misura di dispersione, e come tale va letta: due
    modelli il cui divario e' inferiore alla dispersione dei fold non sono
    distinguibili sotto questo protocollo.
    """
    summary: dict[str, float | str | int] = {}
    if label is not None:
        summary["model"] = label
    for metric in METRICS:
        summary[f"{metric}_mean"] = fold_metrics[metric].mean()
        summary[f"{metric}_std"] = fold_metrics[metric].std(ddof=1)
    summary["n_fit"] = len(fold_metrics)
    summary["fit_seconds_total"] = fold_metrics["fit_seconds"].sum()
    return pd.Series(summary)


def evaluate_holdout(
    estimator, X_train, y_train, X_test, y_test, last_cycle_mask, y_test_raw=None
):
    """Valutazione finale sull'insieme di verifica ufficiale.

    Lo stimatore viene riaddestrato su tutti i motori di addestramento e
    valutato una sola volta. Va invocata a graduatoria gia' chiusa: il suo
    risultato non rientra in nessuna scelta.

    Sono prodotte tre letture. Su tutti i cicli delle traiettorie troncate, che
    e' la lettura estesa. Sul solo ultimo ciclo di ciascuna unita', che e' la
    forma con cui il dataset e' riportato in letteratura. E, se `y_test_raw` e'
    fornito, sull'ultimo ciclo contro il target non censurato, che e' la
    variante in cui la censura si applica all'addestramento ma non alla
    verifica.

    Le tre letture non sono confrontabili fra loro ne' con l'errore in
    cross-validation, perche' riguardano popolazioni di cicli diverse. Il
    troncamento casuale delle traiettorie di verifica ne sposta la composizione
    verso la fase iniziale di vita, dove il target e' appiattito sulla soglia, e
    riduce la variabilita' del target rispetto alle traiettorie complete. Un
    errore assoluto piu' basso sulla verifica che in cross-validation e' quindi
    atteso e non indica un trasferimento migliore: la quota di variabilita'
    spiegata si legge sull'R quadro.
    """
    model = clone(estimator)
    model.fit(X_train, np.asarray(y_train))
    y_pred = model.predict(X_test)
    y_test = np.asarray(y_test)
    mask = np.asarray(last_cycle_mask, dtype=bool)

    result = {f"{k}_all_cycles": v for k, v in regression_metrics(y_test, y_pred).items()}
    result.update(
        {f"{k}_last_cycle": v for k, v in regression_metrics(y_test[mask], y_pred[mask]).items()}
    )
    if y_test_raw is not None:
        raw = np.asarray(y_test_raw)
        result.update(
            {
                f"{k}_last_cycle_raw": v
                for k, v in regression_metrics(raw[mask], y_pred[mask]).items()
            }
        )
    result["n_test_rows"] = len(y_test)
    result["n_test_units"] = int(mask.sum())
    return result, y_pred