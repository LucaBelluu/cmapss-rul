"""Verifica di integrità dei file grezzi C-MAPSS collocati in `data/raw/`.

Ruolo nel progetto
    Controllo di ingresso della catena dati. Esercita il modulo di caricamento
    su tutti e quattro i sottoinsiemi e confronta ciò che viene letto con le
    proprietà attese del dataset. Serve a intercettare un'acquisizione
    incompleta o alterata prima che qualunque analisi vi si appoggi.

Cosa riceve
    I file grezzi in `data/raw/`, non versionati e acquisiti manualmente
    secondo le istruzioni del README.

Cosa produce
    Un resoconto su standard output e un codice di uscita: 0 se tutti i
    controlli passano, 1 se almeno uno fallisce. Non scrive file.

Controlli eseguiti
    1. Numero di righe e numero di unità di ciascun file, confrontati con i
       valori attesi.
    2. Assenza di valori mancanti.
    3. Identificativi delle unità contigui a partire da 1.
    4. Numeri di ciclo di ogni unità consecutivi a partire da 1, cioè
       traiettorie senza interruzioni.
    5. Corrispondenza tra le unità di ciascun file di test e le righe del
       corrispondente file di RUL.
    6. Etichette RUL strettamente positive, coerenti con traiettorie di test
       troncate prima del guasto.

Esecuzione
    Dalla radice della repository:

        python -m scripts.verify_raw_data

    L'invocazione come modulo colloca la radice della repository nel percorso
    di ricerca di Python, che è la condizione per importare `src`.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.data import CYCLE_COL, SUBSETS, UNIT_COL, load_subset

# Valori attesi, fissati sui file effettivamente acquisiti dal repository NASA.
# Sono pin di integrità: uno scostamento segnala un'acquisizione diversa da
# quella su cui il progetto è costruito, non una tolleranza da aggiornare.
EXPECTED = {
    "FD001": {"train_rows": 20631, "train_units": 100, "test_rows": 13096, "test_units": 100},
    "FD002": {"train_rows": 53759, "train_units": 260, "test_rows": 33991, "test_units": 259},
    "FD003": {"train_rows": 24720, "train_units": 100, "test_rows": 16596, "test_units": 100},
    "FD004": {"train_rows": 61249, "train_units": 249, "test_rows": 41214, "test_units": 248},
}


class Report:
    """Raccoglie l'esito dei controlli e tiene traccia dei fallimenti."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, description: str, detail: str = "") -> None:
        if condition:
            print(f"  [ok]   {description}")
        else:
            message = description if not detail else f"{description}: {detail}"
            print(f"  [FAIL] {message}")
            self.failures.append(message)


def cycles_are_consecutive(frame: pd.DataFrame) -> pd.Index:
    """Identificativi delle unità la cui sequenza di cicli non è 1, 2, ..., n."""
    grouped = frame.groupby(UNIT_COL)[CYCLE_COL]
    expected_last = grouped.size()
    observed_first = grouped.min()
    observed_last = grouped.max()
    ok = (observed_first == 1) & (observed_last == expected_last)
    return ok.index[~ok]


def verify_subset(name: str, report: Report) -> None:
    print(f"\n{name}")
    subset = load_subset(name)
    expected = EXPECTED[name]

    for role, frame in (("train", subset.train), ("test", subset.test)):
        units = frame[UNIT_COL].unique()
        n_rows = len(frame)
        n_units = len(units)

        report.check(
            n_rows == expected[f"{role}_rows"],
            f"{role}: righe attese {expected[f'{role}_rows']}",
            f"trovate {n_rows}",
        )
        report.check(
            n_units == expected[f"{role}_units"],
            f"{role}: unità attese {expected[f'{role}_units']}",
            f"trovate {n_units}",
        )

        n_missing = int(frame.isna().sum().sum())
        report.check(n_missing == 0, f"{role}: nessun valore mancante", f"{n_missing} valori")

        contiguous = list(units) == list(range(1, n_units + 1))
        report.check(contiguous, f"{role}: identificativi delle unità contigui da 1")

        broken = cycles_are_consecutive(frame)
        report.check(
            len(broken) == 0,
            f"{role}: cicli consecutivi da 1 in ogni unità",
            f"{len(broken)} unità irregolari, prime: {list(broken[:5])}",
        )

    rul = subset.rul
    report.check(
        len(rul) == expected["test_units"],
        f"rul: una etichetta per unità di test ({expected['test_units']})",
        f"trovate {len(rul)}",
    )
    report.check(
        list(rul.index) == list(subset.test[UNIT_COL].unique()),
        "rul: indice allineato alle unità del file di test",
    )
    report.check(
        bool((rul > 0).all()),
        "rul: etichette strettamente positive",
        f"minimo {int(rul.min())}",
    )


def main() -> int:
    report = Report()
    for name in SUBSETS:
        verify_subset(name, report)

    print()
    if report.failures:
        print(f"Verifica fallita: {len(report.failures)} controlli non superati.")
        return 1

    print("Verifica superata: tutti i controlli sono stati superati.")
    return 0


if __name__ == "__main__":
    sys.exit(main())