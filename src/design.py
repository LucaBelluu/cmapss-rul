"""Costruzione della matrice di progetto per un sottoinsieme C-MAPSS.

Ruolo nel progetto
    Terzo stadio della catena dati, fra la costruzione del target e la
    valutazione. Trasforma le traiettorie caricate da `src.data` e arricchite da
    `src.target` nelle strutture su cui opera il protocollo: una matrice di
    variabili esplicative, un vettore target, un vettore di identificativi di
    unita' per il partizionamento.

Cosa riceve
    Il nome di un sottoinsieme ("FD001", "FD003") e la soglia di censura.

Cosa produce
    Una struttura `Design` che tiene insieme la parte di addestramento e la
    parte di verifica ufficiale dello stesso sottoinsieme, gia' allineate sulle
    stesse colonne e nello stesso ordine.

Variabili esplicative
    Numero di ciclo, impostazioni operative e letture dei sensori non costanti.

    Il numero di ciclo e' incluso. Non e' una fuga di informazione: il numero di
    cicli percorsi e' noto al momento della predizione anche su una traiettoria
    troncata. Va pero' tenuto presente che su traiettorie che arrivano tutte al
    guasto la vita utile residua e' per costruzione la differenza fra durata e
    ciclo corrente, quindi una parte della capacita' predittiva di qualunque
    modello proviene da un conteggio e non dalla lettura del degrado. Questa
    relazione e' esatta sulle traiettorie complete e non lo e' su quelle
    troncate, dove il punto di interruzione e' casuale: il contributo del
    numero di ciclo si trasferisce quindi solo in parte dall'insieme di
    addestramento a quello di verifica. Per rendere misurabile questa parte il
    confronto include una baseline che usa il solo numero di ciclo, rispetto
    alla quale si legge il guadagno dei modelli che usano i sensori.

    Le colonne costanti vengono rimosse: portano zero informazione e la loro
    standardizzazione e' una divisione per una quantita' nulla. Il criterio e'
    il numero di valori distinti, esatto per costruzione, e non la deviazione
    standard, che su una colonna di valori identici restituisce un residuo di
    arrotondamento non nullo. Le costanti sono determinate sulle sole
    traiettorie di addestramento del sottoinsieme, e sono una proprieta'
    strutturale del sensore in quel regime operativo: non dipendono dal target,
    quindi determinarle sull'intera parte di addestramento non introduce
    informazione proveniente dalle partizioni di verifica.

    L'elenco atteso e' verificato contro un valore cablato: uno scostamento
    indica dati diversi da quelli su cui il progetto e' costruito, ed e' un
    errore, non una variazione da assorbire.

Parte di verifica ufficiale
    Le traiettorie di verifica sono troncate e ciascuna unita' ha una sola
    etichetta di vita utile residua, riferita all'ultimo ciclo osservato. Da
    quella si ricava il target a ogni ciclo, quindi la parte di verifica e'
    utilizzabile per intero e non soltanto sull'ultimo ciclo. La maschera
    `last_cycle` individua le righe finali di ciascuna unita' e permette la
    lettura ristretta con cui il dataset e' riportato in letteratura.

    Il troncamento e' casuale, quindi la composizione delle traiettorie di
    verifica e' spostata verso la fase iniziale di vita rispetto a quelle di
    addestramento, che arrivano tutte al guasto. La quota di righe al valore di
    soglia e' percio' piu' alta e la variabilita' del target piu' bassa: i
    valori assoluti delle metriche calcolate sulle due parti non sono
    confrontabili fra loro. `describe` riporta entrambe le quantita' proprio
    perche' la differenza sia leggibile e non venga scambiata per un effetto dei
    modelli.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import (
    CYCLE_COL,
    SENSOR_COLS,
    SETTING_COLS,
    UNIT_COL,
    load_subset,
)
from src.target import (
    RUL_CAP,
    RUL_COL,
    add_censored_rul,
    add_rul_from_labels,
    censor,
)

# Sottoinsiemi nel perimetro sperimentale. Tengono fermo il regime operativo e
# fanno variare il solo numero di modi di guasto.
SUBSETS_IN_SCOPE = ("FD001", "FD003")

# Colonne costanti attese, verificate in fase di esplorazione. Sono pin di
# integrita': il codice le ricalcola e confronta.
EXPECTED_CONSTANTS = {
    "FD001": [
        "setting_3",
        "sensor_01",
        "sensor_05",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ],
    "FD003": [
        "setting_3",
        "sensor_01",
        "sensor_05",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ],
}

CANDIDATE_COLS = [CYCLE_COL] + SETTING_COLS + SENSOR_COLS


@dataclass(frozen=True)
class Design:
    """Matrice di progetto e target di un sottoinsieme, parte di addestramento e
    parte di verifica ufficiale.

    X_train, y_train, groups_train
        Traiettorie complete fino al guasto. `groups_train` e' l'identificativo
        del motore di ciascuna riga e viene usato per il partizionamento.
    X_test, y_test, groups_test
        Traiettorie troncate del file di verifica ufficiale, con target
        ricostruito dalle etichette e censurato alla stessa soglia.
    y_test_raw
        Lo stesso target senza censura. Serve unicamente alla lettura di
        raffronto con la letteratura, dove le predizioni sono valutate contro la
        vita utile residua effettiva. Non entra in nessuna selezione.
    last_cycle
        Maschera booleana sulle righe di verifica: vera sull'ultimo ciclo
        osservato di ciascuna unita'.
    """

    subset: str
    cap: int | None
    features: list[str]
    dropped: list[str]
    X_train: pd.DataFrame = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    groups_train: np.ndarray = field(repr=False)
    X_test: pd.DataFrame = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    y_test_raw: np.ndarray = field(repr=False)
    groups_test: np.ndarray = field(repr=False)
    last_cycle: np.ndarray = field(repr=False)

    @property
    def n_units_train(self) -> int:
        return len(np.unique(self.groups_train))

    @property
    def n_units_test(self) -> int:
        return len(np.unique(self.groups_test))


def constant_columns(frame: pd.DataFrame, columns=None) -> list[str]:
    """Colonne con un solo valore distinto.

    Il criterio e' il conteggio dei valori distinti e non la deviazione
    standard: su una colonna di valori identici la deviazione standard calcolata
    numericamente vale circa 1e-13 e il confronto con zero fallisce.
    """
    columns = list(frame.columns) if columns is None else list(columns)
    return [c for c in columns if frame[c].nunique(dropna=False) == 1]


def build_design(
    subset: str,
    *,
    cap: int | None = RUL_CAP,
    raw_dir=None,
    check_constants: bool = True,
) -> Design:
    """Costruisce la matrice di progetto di un sottoinsieme.

    cap
        Soglia di censura del target. `None` disattiva la censura ed e' il modo
        in cui si esegue il controllo di sensibilita' sul target lineare.
    check_constants
        Se vero, confronta le colonne costanti trovate con quelle attese e
        solleva un errore in caso di scostamento.
    """
    data = load_subset(subset, raw_dir)

    train = add_censored_rul(data.train, cap=cap)

    # Il target di verifica viene costruito prima senza censura e censurato
    # dopo, cosi' da conservare entrambe le versioni: la censurata e' quella su
    # cui il progetto valuta, la non censurata serve alla sola lettura di
    # raffronto con la letteratura.
    test = add_rul_from_labels(data.test, data.rul)
    y_test_raw = test[RUL_COL].to_numpy()
    test[RUL_COL] = censor(test[RUL_COL], cap).astype("int32")

    dropped = constant_columns(train, CANDIDATE_COLS)
    if check_constants and subset in EXPECTED_CONSTANTS:
        expected = EXPECTED_CONSTANTS[subset]
        if sorted(dropped) != sorted(expected):
            raise AssertionError(
                f"{subset}: colonne costanti {sorted(dropped)}, attese {sorted(expected)}."
            )

    features = [c for c in CANDIDATE_COLS if c not in dropped]

    # La maschera dell'ultimo ciclo va calcolata prima di ridurre alle sole
    # colonne esplicative, perche' usa l'identificativo di unita' e il ciclo.
    last_cycle_flag = (
        test[CYCLE_COL] == test.groupby(UNIT_COL)[CYCLE_COL].transform("max")
    ).to_numpy()

    # Controllo di integrita' sul target di verifica: sull'ultimo ciclo di ogni
    # unita' deve coincidere con l'etichetta, censurata alla stessa soglia. Un
    # disallineamento posizionale fra etichette e unita' produrrebbe un target
    # quasi costante e facile da predire, cioe' un risultato migliore del vero
    # senza che nulla segnali l'errore.
    expected_at_end = (
        censor(data.rul, cap).astype("int32").sort_index().to_numpy()
    )
    observed_at_end = (
        test.loc[last_cycle_flag, [UNIT_COL, RUL_COL]]
        .sort_values(UNIT_COL)[RUL_COL]
        .to_numpy()
    )
    if not np.array_equal(observed_at_end, expected_at_end):
        raise AssertionError(
            f"{subset}: il target di verifica sull'ultimo ciclo non coincide "
            f"con le etichette RUL."
        )

    return Design(
        subset=subset,
        cap=cap,
        features=features,
        dropped=dropped,
        X_train=train[features].reset_index(drop=True),
        y_train=train[RUL_COL].to_numpy(),
        groups_train=train[UNIT_COL].to_numpy(),
        X_test=test[features].reset_index(drop=True),
        y_test=test[RUL_COL].to_numpy(),
        y_test_raw=y_test_raw,
        groups_test=test[UNIT_COL].to_numpy(),
        last_cycle=last_cycle_flag,
    )


def describe(design: Design) -> pd.Series:
    """Riepilogo numerico della matrice, per la registrazione degli esperimenti."""
    censored_train = (
        float((design.y_train == design.cap).mean()) if design.cap is not None else 0.0
    )
    censored_test = (
        float((design.y_test == design.cap).mean()) if design.cap is not None else 0.0
    )
    return pd.Series(
        {
            "subset": design.subset,
            "cap": "none" if design.cap is None else design.cap,
            "n_features": len(design.features),
            "n_dropped": len(design.dropped),
            "train_rows": len(design.X_train),
            "train_units": design.n_units_train,
            "test_rows": len(design.X_test),
            "test_units": design.n_units_test,
            "y_train_mean": float(design.y_train.mean()),
            "y_train_std": float(design.y_train.std(ddof=1)),
            "y_train_censored_share": censored_train,
            "y_test_mean": float(design.y_test.mean()),
            "y_test_std": float(design.y_test.std(ddof=1)),
            "y_test_censored_share": censored_test,
        }
    )