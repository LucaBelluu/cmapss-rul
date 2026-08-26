"""Costruzione della variabile target (vita utile residua) sulle traiettorie.

Ruolo nel progetto
    Trasforma le traiettorie caricate da `src.data` aggiungendo la variabile da
    predire. È separato dal caricamento perché il target non è un dato letto
    dai file: è una quantità derivata, e la sua definizione è una scelta del
    progetto.

Cosa riceve
    Un DataFrame di traiettorie nel formato restituito da `src.data`, con le
    colonne `unit` e `cycle`. Per le traiettorie di test serve anche la serie di
    etichette restituita da `src.data.load_rul`.

Cosa produce
    Lo stesso DataFrame con in più la colonna `rul`.

Definizione adottata
    Nei file di training ogni traiettoria arriva al guasto, quindi l'ultimo
    ciclo osservato di un'unità è il ciclo del guasto e la vita utile residua a
    un ciclo qualsiasi è la differenza tra i due. Nei file di test le
    traiettorie sono troncate e la vita utile residua all'ultimo ciclo è nota
    solo dal file di etichette: le due situazioni hanno due funzioni distinte,
    che calcolano la stessa quantità a partire da informazioni diverse.

    Sopra questa definizione viene applicata una censura a soglia: la vita utile
    residua è troncata a un valore massimo, oltre il quale il target resta
    costante. La ragione è che nella prima parte della vita di un motore il
    degrado non è ancora osservabile dai sensori, e le letture di due unità con
    vite residue molto diverse sono in quella fase indistinguibili. Senza
    censura il target contiene una componente che nessun modello può predire, e
    che pesa in modo sproporzionato in una metrica quadratica perché ricade sui
    valori più grandi.

    La soglia adottata nel progetto è di 125 cicli. È inferiore alla durata
    della traiettoria più breve di entrambi i sottoinsiemi impiegati (128 cicli
    in FD001, 145 in FD003), quindi ogni unità contribuisce sia con una fase
    censurata sia con una fase di degrado e nessuna traiettoria risulta
    interamente costante.

    La soglia è un'ipotesi di modellazione, non una quantità misurata: i valori
    assoluti delle metriche dipendono da essa. È fissata a priori e non viene
    selezionata sui risultati. Una selezione della soglia per cross-validation
    non sarebbe legittima: abbassando la soglia si restringe l'intervallo dei
    valori da predire e l'errore quadratico medio cala per costruzione, quindi
    il confronto premierebbe sempre la soglia più bassa a prescindere dalla
    qualità del modello.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import CYCLE_COL, UNIT_COL

RUL_COL = "rul"

# Soglia di censura del target, in cicli. Valore di progetto, fissato a priori.
RUL_CAP = 125


def add_linear_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna `rul` non censurata a traiettorie complete fino al guasto.

    Il DataFrame di ingresso non viene modificato: la funzione restituisce una
    copia. Il calcolo avviene per unità, come differenza tra l'ultimo ciclo
    dell'unità e il ciclo della riga.
    """
    _require_columns(frame)

    result = frame.copy()
    last_cycle = result.groupby(UNIT_COL)[CYCLE_COL].transform("max")
    result[RUL_COL] = (last_cycle - result[CYCLE_COL]).astype("int32")

    return result


def add_rul_from_labels(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Aggiunge la colonna `rul` non censurata a traiettorie troncate.

    Per le unità di test la vita utile residua all'ultimo ciclo osservato è
    fornita dal file di etichette. La vita utile residua a un ciclo precedente
    si ottiene sommando i cicli che mancano alla fine della traiettoria
    osservata.

    labels
        Serie indicizzata per identificativo di unità, nel formato restituito da
        `src.data.load_rul`.
    """
    _require_columns(frame)

    missing = set(frame[UNIT_COL].unique()) - set(labels.index)
    if missing:
        raise ValueError(
            f"Etichette assenti per {len(missing)} unità, ad esempio "
            f"{sorted(missing)[:5]}."
        )

    result = frame.copy()
    last_cycle = result.groupby(UNIT_COL)[CYCLE_COL].transform("max")
    residual_at_end = result[UNIT_COL].map(labels)
    result[RUL_COL] = (residual_at_end + last_cycle - result[CYCLE_COL]).astype("int32")

    return result


def censor(values: pd.Series | np.ndarray, cap: int = RUL_CAP) -> pd.Series | np.ndarray:
    """Tronca la vita utile residua alla soglia indicata.

    Applicata a una colonna già calcolata, in modo che la censura resti una
    trasformazione visibile e separata dalla definizione del target. Passando
    `cap=None` la funzione restituisce i valori invariati, che è il modo in cui
    si esegue il controllo di sensibilità sul target non censurato.
    """
    if cap is None:
        return values
    if isinstance(values, pd.Series):
        return values.clip(upper=cap)
    return np.minimum(values, cap)


def add_censored_rul(frame: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Aggiunge la colonna `rul` censurata a traiettorie complete fino al guasto."""
    result = add_linear_rul(frame)
    result[RUL_COL] = censor(result[RUL_COL], cap).astype("int32")
    return result


def add_censored_rul_from_labels(
    frame: pd.DataFrame, labels: pd.Series, cap: int = RUL_CAP
) -> pd.DataFrame:
    """Aggiunge la colonna `rul` censurata a traiettorie troncate."""
    result = add_rul_from_labels(frame, labels)
    result[RUL_COL] = censor(result[RUL_COL], cap).astype("int32")
    return result


def _require_columns(frame: pd.DataFrame) -> None:
    missing = {UNIT_COL, CYCLE_COL} - set(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel DataFrame: {sorted(missing)}.")