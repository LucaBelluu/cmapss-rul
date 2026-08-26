"""Statistiche descrittive dei sottoinsiemi C-MAPSS.

Ruolo nel progetto
    Contiene il calcolo delle grandezze su cui si basa l'esplorazione. Ogni
    funzione riceve dati già caricati e restituisce un DataFrame in formato
    lungo, senza scrivere su disco e senza produrre grafici: la scrittura degli
    artefatti sta in `scripts/run_exploration.py`, la loro rappresentazione nel
    notebook di esplorazione.

Cosa riceve
    Traiettorie nel formato restituito da `src.data`.

Cosa produce
    DataFrame con una colonna `subset` che identifica il sottoinsieme di
    provenienza, in modo che i risultati dei quattro sottoinsiemi si
    concatenino senza ambiguità.

Nota sulle correlazioni con il target
    Le correlazioni calcolate qui sono descrittive e servono a capire quali
    sensori portano segnale. Non costituiscono una selezione di variabili: una
    selezione basata su una statistica calcolata sull'intero insieme di
    addestramento userebbe informazione proveniente anche dalle porzioni su cui
    poi si misura, e va invece eseguita dentro il flusso di validazione. Le
    correlazioni sono inoltre calcolate accorpando le righe di tutte le unità,
    quindi ignorano il raggruppamento per motore e vanno lette come indicazione
    di massima.
"""

from __future__ import annotations

import pandas as pd

from src.data import CYCLE_COL, SENSOR_COLS, SETTING_COLS, UNIT_COL
from src.target import RUL_CAP, RUL_COL, add_linear_rul

# Le impostazioni operative sono registrate con rumore attorno a pochi valori
# nominali. L'arrotondamento riporta ciascuna al proprio valore nominale e
# rende identificabili i regimi di volo distinti. Le cifre sono scelte in base
# alla scala delle tre impostazioni: quota e angolo della manetta hanno scala
# unitaria, il numero di Mach ha scala centesimale.
CONDITION_ROUNDING = {"setting_1": 0, "setting_2": 2, "setting_3": 0}
CONDITION_COL = "condition"


def constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Colonne che assumono un unico valore.

    Il criterio è il numero di valori distinti e non la deviazione standard: su
    una colonna di valori identici la deviazione standard calcolata
    numericamente non è esattamente nulla ma un residuo di arrotondamento
    dell'ordine di 1e-13, e il confronto con zero classificherebbe come
    variabile una colonna costante.
    """
    return [c for c in columns if frame[c].nunique(dropna=False) <= 1]


def trajectory_lengths(frame: pd.DataFrame, subset: str, split: str) -> pd.DataFrame:
    """Numero di cicli osservati per ciascuna unità.

    Nei file di training corrisponde alla durata di vita del motore, nei file di
    test alla lunghezza della porzione osservata prima del troncamento.
    """
    lengths = frame.groupby(UNIT_COL)[CYCLE_COL].max().rename("length").reset_index()
    lengths.insert(0, "split", split)
    lengths.insert(0, "subset", subset)
    return lengths


def operating_conditions(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Regimi di funzionamento distinti, con la loro frequenza.

    Ogni regime è la terna delle impostazioni operative arrotondate ai valori
    nominali. Il numero di regimi distinti è ciò che separa i sottoinsiemi a
    condizione singola da quelli a condizioni multiple.
    """
    labelled = label_conditions(frame)

    counts = (
        labelled.groupby(SETTING_COLS + [CONDITION_COL], as_index=False)
        .agg(n_rows=(CYCLE_COL, "size"), n_units=(UNIT_COL, "nunique"))
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )
    counts.insert(0, "subset", subset)
    return counts


def label_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le impostazioni arrotondate e un'etichetta testuale di regime."""
    labelled = frame.copy()
    for column, decimals in CONDITION_ROUNDING.items():
        # L'aggiunta di zero elimina lo zero negativo prodotto
        # dall'arrotondamento di valori leggermente inferiori a zero, che
        # altrimenti genererebbe due regimi distinti indistinguibili a stampa.
        labelled[column] = labelled[column].round(decimals) + 0.0

    labelled[CONDITION_COL] = (
        labelled[SETTING_COLS].astype(str).agg(" | ".join, axis=1)
    )
    return labelled


def variable_summary(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Statistiche di base di impostazioni operative e sensori.

    Il numero di valori distinti affianca la deviazione standard perché
    distingue due situazioni diverse: una variabile costante e una variabile che
    assume pochi livelli discreti. La seconda porta informazione, la prima no.
    """
    columns = SETTING_COLS + SENSOR_COLS
    constants = set(constant_columns(frame, columns))

    summary = pd.DataFrame(
        {
            "variable": columns,
            "mean": [frame[c].mean() for c in columns],
            "std": [frame[c].std() for c in columns],
            "min": [frame[c].min() for c in columns],
            "max": [frame[c].max() for c in columns],
            "n_unique": [frame[c].nunique() for c in columns],
        }
    )
    summary["constant"] = summary["variable"].isin(constants)
    summary.insert(0, "subset", subset)
    return summary


def variable_summary_by_condition(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Le stesse statistiche, calcolate dentro ciascun regime di funzionamento.

    Nei sottoinsiemi a condizioni multiple una variabile può risultare molto
    variabile complessivamente e quasi costante dentro ogni singolo regime: in
    quel caso la variabilità osservata è dovuta al regime di volo e non al
    degrado. Il confronto tra questa tabella e quella complessiva rende visibile
    la differenza.
    """
    labelled = label_conditions(frame)

    rows = []
    for condition, group in labelled.groupby(CONDITION_COL):
        block = variable_summary(group, subset)
        block.insert(1, CONDITION_COL, condition)
        block["n_rows"] = len(group)
        rows.append(block)

    return pd.concat(rows, ignore_index=True)


def target_correlations(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Correlazione di ciascuna variabile con la vita utile residua.

    Sono riportati sia il coefficiente di Pearson, che misura la componente
    lineare della relazione, sia quello di Spearman, che coglie una relazione
    monotona anche non lineare. La differenza tra i due segnala relazioni non
    lineari e riguarda direttamente la scelta tra modelli lineari e modelli che
    non lo sono.

    Per le variabili costanti la correlazione non è definita e viene riportata
    come valore mancante, senza tentarne il calcolo.
    """
    with_target = add_linear_rul(frame)
    columns = SETTING_COLS + SENSOR_COLS
    constants = set(constant_columns(frame, columns))

    records = []
    for column in columns:
        if column in constants:
            records.append({"variable": column, "pearson": None, "spearman": None})
            continue
        records.append(
            {
                "variable": column,
                "pearson": with_target[column].corr(with_target[RUL_COL]),
                "spearman": with_target[column].corr(
                    with_target[RUL_COL], method="spearman"
                ),
            }
        )

    correlations = pd.DataFrame.from_records(records)
    correlations["abs_pearson"] = correlations["pearson"].abs()
    correlations["constant"] = correlations["variable"].isin(constants)
    correlations.insert(0, "subset", subset)
    return correlations


def target_correlations_by_phase(
    frame: pd.DataFrame, subset: str, cap: int = RUL_CAP
) -> pd.DataFrame:
    """Correlazione con la RUL calcolata separatamente nelle due fasi di vita.

    Le righe sono divise in due gruppi in base alla vita utile residua: quelle
    con residuo superiore alla soglia di censura e quelle con residuo inferiore
    o uguale. La censura del target assume che nella prima fase il degrado non
    sia ancora osservabile dai sensori, cioè che le letture non varino al
    variare della vita residua. La tabella misura direttamente questa
    assunzione: una correlazione prossima a zero sopra la soglia e marcata sotto
    la soglia la conferma, valori simili nelle due fasi la smentiscono.

    La divisione usa la vita utile residua non censurata, perché serve a
    caratterizzare il dato e non il target trasformato.
    """
    with_target = add_linear_rul(frame)
    columns = SETTING_COLS + SENSOR_COLS
    constants = set(constant_columns(frame, columns))

    phases = {
        "oltre_soglia": with_target[RUL_COL] > cap,
        "entro_soglia": with_target[RUL_COL] <= cap,
    }

    records = []
    for phase, mask in phases.items():
        block = with_target[mask]
        for column in columns:
            degenerate = column in constants or block[column].nunique() <= 1
            records.append(
                {
                    "subset": subset,
                    "phase": phase,
                    "cap": cap,
                    "variable": column,
                    "n_rows": len(block),
                    "pearson": None if degenerate else block[column].corr(block[RUL_COL]),
                    "spearman": None
                    if degenerate
                    else block[column].corr(block[RUL_COL], method="spearman"),
                }
            )

    correlations = pd.DataFrame.from_records(records)
    correlations["abs_pearson"] = correlations["pearson"].abs()
    return correlations


def variable_correlation_matrix(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Matrice di correlazione tra sensori, in formato lungo.

    Il formato lungo (una riga per coppia) è usato al posto della matrice
    quadrata perché consente di concatenare i quattro sottoinsiemi in un unico
    file senza perdere l'identificazione della provenienza. Le coppie che
    coinvolgono un sensore costante compaiono con valore mancante.
    """
    constants = set(constant_columns(frame, SENSOR_COLS))
    varying = [c for c in SENSOR_COLS if c not in constants]

    matrix = frame[varying].corr().reindex(index=SENSOR_COLS, columns=SENSOR_COLS)
    long_form = matrix.stack().rename("pearson").reset_index()
    long_form.columns = ["variable_a", "variable_b", "pearson"]
    long_form.insert(0, "subset", subset)
    return long_form


def sensor_traces(frame: pd.DataFrame, subset: str, n_units: int = 5) -> pd.DataFrame:
    """Traiettorie complete di poche unità, per l'ispezione visiva del degrado.

    Serve a mostrare l'andamento dei sensori lungo la vita di un motore. Le
    unità sono le prime `n_units` per identificativo, scelte senza guardare i
    dati per non selezionare i casi che raccontano meglio la storia.
    """
    with_target = add_linear_rul(frame)
    units = sorted(with_target[UNIT_COL].unique())[:n_units]
    traces = with_target[with_target[UNIT_COL].isin(units)].copy()
    traces.insert(0, "subset", subset)
    return traces