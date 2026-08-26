"""Caricamento dei file grezzi del dataset NASA C-MAPSS.

Ruolo nel progetto
    Primo stadio della catena dati e unico punto in cui i file di
    `data/raw/` vengono letti. Ogni altro modulo del progetto lavora sulle
    strutture restituite da qui e non riapre i file originali.

Cosa riceve
    I file di testo distribuiti dal NASA Prognostics Data Repository,
    collocati senza modifiche in `data/raw/`. Per ciascuno dei quattro
    sottoinsiemi (FD001, FD002, FD003, FD004) sono presenti un file di
    training, un file di test e un file di etichette RUL.

Cosa produce
    DataFrame con colonne nominate e tipizzate, una serie di etichette RUL
    indicizzata per unità, e la struttura `CmapssSubset` che tiene insieme i
    tre file di uno stesso sottoinsieme.

Formato dei file grezzi
    Ogni riga di un file di training o di test è un ciclo di funzionamento di
    un motore e contiene 26 valori separati da spazi: identificativo
    dell'unità, numero di ciclo, 3 impostazioni operative, 21 letture di
    sensori. I file non hanno riga di intestazione. Le righe di uno stesso
    motore sono consecutive e ordinate per numero di ciclo crescente.

    Nei file di training ogni traiettoria arriva al guasto, quindi la vita
    utile residua a ogni ciclo si ricava per differenza dall'ultimo ciclo
    della stessa unità. Nei file di test le traiettorie sono troncate prima
    del guasto e il file di RUL corrispondente riporta un solo valore per
    unità, cioè la vita utile residua all'ultimo ciclo osservato. Il file di
    RUL non contiene identificativi: l'associazione con le unità è
    posizionale e segue l'ordine crescente degli identificativi.

Nomi dei sensori
    Le 21 letture sono numerate da 01 a 21 nell'ordine in cui compaiono nei
    file. Le sigle fisiche corrispondenti si trovano nella documentazione
    originale conservata in `data/raw/`. La numerazione posizionale è usata
    perché verificabile direttamente sul dato, mentre la corrispondenza con
    le sigle dipende da una fonte esterna al file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Radice della repository, ricavata dalla posizione di questo file: `src/` sta
# un livello sotto la radice. Ricavarla dal file e non dalla directory di
# lavoro rende il caricamento indipendente da dove viene lanciato il processo,
# in particolare dai notebook che risiedono in `notebooks/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

UNIT_COL = "unit"
CYCLE_COL = "cycle"
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i:02d}" for i in range(1, 22)]
COLUMN_NAMES = [UNIT_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS

N_COLUMNS = len(COLUMN_NAMES)

RUL_COL = "rul_last_cycle"


@dataclass(frozen=True)
class CmapssSubset:
    """I tre file di un sottoinsieme FD00X, tenuti insieme.

    name
        Identificativo del sottoinsieme, ad esempio "FD001".
    train
        Traiettorie complete fino al guasto, una riga per ciclo.
    test
        Traiettorie troncate prima del guasto, una riga per ciclo.
    rul
        Vita utile residua all'ultimo ciclo osservato di ciascuna unità di
        test, indicizzata per identificativo di unità.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series


def _resolve(raw_dir: Path | str | None) -> Path:
    return RAW_DIR if raw_dir is None else Path(raw_dir)


def _read_cycles(path: Path) -> pd.DataFrame:
    """Legge un file di traiettorie e restituisce un DataFrame nominato.

    La lettura avviene senza passare i nomi delle colonne, per poter
    verificare quante colonne il file contiene davvero prima di assegnarli:
    associare i nomi in fase di lettura maschererebbe un file con un numero di
    campi diverso da quello atteso.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}. I dati grezzi non sono versionati e "
            f"vanno collocati in {RAW_DIR} seguendo le istruzioni di "
            f"acquisizione documentate nel README."
        )

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # Le righe dei file originali terminano con spazi. A seconda della
    # versione di pandas questo produce una colonna finale interamente vuota,
    # che non fa parte del dato e viene rimossa prima del controllo.
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] != N_COLUMNS:
        raise ValueError(
            f"{path.name}: attese {N_COLUMNS} colonne, trovate {frame.shape[1]}."
        )

    frame.columns = COLUMN_NAMES

    # Identificativo e numero di ciclo sono conteggi e vengono tipizzati come
    # interi: lasciarli in virgola mobile renderebbe fragili i raggruppamenti
    # per unità e i confronti tra numeri di ciclo.
    frame[UNIT_COL] = frame[UNIT_COL].astype("int32")
    frame[CYCLE_COL] = frame[CYCLE_COL].astype("int32")
    frame[SETTING_COLS + SENSOR_COLS] = frame[SETTING_COLS + SENSOR_COLS].astype(
        "float64"
    )

    return frame


def load_train(subset: str, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Traiettorie di training del sottoinsieme indicato, complete fino al guasto."""
    _check_subset(subset)
    return _read_cycles(_resolve(raw_dir) / f"train_{subset}.txt")


def load_test(subset: str, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Traiettorie di test del sottoinsieme indicato, troncate prima del guasto."""
    _check_subset(subset)
    return _read_cycles(_resolve(raw_dir) / f"test_{subset}.txt")


def load_rul(subset: str, raw_dir: Path | str | None = None) -> pd.Series:
    """Etichette RUL delle unità di test, indicizzate per identificativo di unità.

    Il file contiene un valore per riga e nessun identificativo. L'indice viene
    ricostruito come 1..N seguendo l'ordine delle righe, che corrisponde
    all'ordine crescente degli identificativi delle unità di test. La
    corrispondenza va verificata contro il file di test prima dell'uso.
    """
    _check_subset(subset)
    path = _resolve(raw_dir) / f"RUL_{subset}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}. I dati grezzi non sono versionati e "
            f"vanno collocati in {RAW_DIR} seguendo le istruzioni di "
            f"acquisizione documentate nel README."
        )

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] != 1:
        raise ValueError(
            f"{path.name}: attesa 1 colonna, trovate {frame.shape[1]}."
        )

    values = frame.iloc[:, 0].astype("int32")
    values.index = pd.RangeIndex(start=1, stop=len(values) + 1, name=UNIT_COL)
    values.name = RUL_COL

    return values


def load_subset(subset: str, raw_dir: Path | str | None = None) -> CmapssSubset:
    """Training, test ed etichette RUL di un singolo sottoinsieme."""
    _check_subset(subset)
    return CmapssSubset(
        name=subset,
        train=load_train(subset, raw_dir),
        test=load_test(subset, raw_dir),
        rul=load_rul(subset, raw_dir),
    )


def load_all(raw_dir: Path | str | None = None) -> dict[str, CmapssSubset]:
    """Tutti e quattro i sottoinsiemi, in un dizionario indicizzato per nome."""
    return {name: load_subset(name, raw_dir) for name in SUBSETS}


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(
            f"Sottoinsieme non riconosciuto: {subset!r}. Valori ammessi: "
            f"{', '.join(SUBSETS)}."
        )