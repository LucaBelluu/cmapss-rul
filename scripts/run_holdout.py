"""Lettura dell'insieme di verifica ufficiale sui modelli selezionati.

Ruolo nel progetto
    Ultimo stadio del confronto. Ricostruisce dagli artefatti della graduatoria
    ogni modello selezionato, lo riaddestra sull'intera parte di addestramento
    del sottoinsieme e lo valuta una sola volta sull'insieme di verifica
    ufficiale. È l'unico punto del progetto in cui `test_FD00X.txt` e
    `RUL_FD00X.txt` entrano in una misura di prestazione.

Cosa riceve
    La graduatoria complessiva in `experiments/final/{SUBSET}_ranking.csv`, il
    registro dei modelli e i dati grezzi attraverso la catena `src.data`,
    `src.target`, `src.design`. Nessun argomento obbligatorio.

Cosa produce
    In `experiments/final/`, per ciascun sottoinsieme:

    - `{SUBSET}_holdout.csv`, una riga per modello con il punteggio in
      cross-validation e le tre letture dell'insieme di verifica;
    - `{SUBSET}_holdout_predictions.csv`, le predizioni su ogni riga di
      verifica, in forma lunga, che sono il materiale delle figure di
      residuo del notebook di sintesi;
    - `{SUBSET}_holdout_agreement.csv`, la correlazione di rango fra la
      graduatoria in cross-validation e ciascuna lettura della verifica.

Regola di lettura, fissata prima che questo file esistesse
    La graduatoria del progetto è quella in cross-validation. L'insieme di
    verifica misura il trasferimento fuori campione e non riordina: ha un solo
    punteggio per modello, senza misura di variabilità, e ordinare su di esso
    significherebbe ordinare su un numero di cui non si conosce l'incertezza.
    La tabella prodotta conserva perciò l'ordine della graduatoria, e le
    colonne di rango sono affiancate perché lo spostamento sia leggibile senza
    che la tabella venga riordinata.

    Vengono letti tutti i modelli della graduatoria e le due baseline, non i
    soli migliori, così che il confronto fuori campione sia disponibile per
    l'intera tabella e non per la parte che conviene.

    Le tre letture (tutti i cicli delle traiettorie troncate, solo ultimo ciclo,
    ultimo ciclo contro target non censurato) non sono confrontabili fra loro
    né con l'errore in cross-validation, perché riguardano popolazioni di
    cicli diverse. Il troncamento casuale sposta la composizione della verifica
    verso la fase iniziale di vita, dove il target è appiattito sulla soglia:
    un errore assoluto più basso sulla verifica è atteso e non indica un
    trasferimento migliore.

Controllo di fedeltà incorporato
    Le due baseline e la regressione lineare multipla sono state valutate
    sull'insieme di verifica in fase di convalida del protocollo, e i loro
    punteggi sono registrati. Sono modelli deterministici e senza
    iperparametri: devono riprodursi. Uno scostamento significa che la catena
    dati è cambiata da allora, e in quel caso nessuna delle altre righe è
    interpretabile. Il controllo blocca l'esecuzione, ma solo dopo aver scritto
    gli artefatti: una corsa lunga che fallisce l'ultimo controllo non va persa.

Esecuzioni parziali
    Con `--models` la corsa è parziale e non scrive su disco. Una
    riesecuzione limitata a un sottoinsieme dei modelli che sovrascrivesse la
    tabella completa lascerebbe la cartella apparentemente intatta e la tabella
    incompleta, che è un difetto già osservato sugli artefatti di un blocco e
    che non lascia traccia perché gli artefatti non sono versionati.

Come si lancia
    python -m scripts.run_holdout --subsets FD001 --models ols baseline_costante baseline_solo_ciclo
    python -m scripts.run_holdout
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.final import BASELINE_KEYS
from src.protocol import evaluate_holdout
from src.selected import load_ranking, rebuild_all
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final"

# Punteggi registrati in fase di convalida del protocollo, sui soli modelli
# deterministici e senza iperparametri. Sono arrotondati al centesimo di ciclo,
# che è la precisione con cui sono stati registrati, e la tolleranza del
# confronto è fissata di conseguenza: assorbe l'arrotondamento e nient'altro.
REFERENCE_HOLDOUT = {
    ("FD001", "baseline_costante"): (35.34, 41.94),
    ("FD001", "baseline_solo_ciclo"): (23.69, 32.25),
    ("FD001", "ols"): (19.07, 21.45),
    ("FD003", "baseline_costante"): (31.37, 43.70),
    ("FD003", "baseline_solo_ciclo"): (26.03, 36.80),
    ("FD003", "ols"): (17.96, 21.44),
}
REFERENCE_TOLERANCE = 5e-3

READINGS = {
    "rmse_all_cycles": "tutti i cicli",
    "rmse_last_cycle": "solo ultimo ciclo",
    "rmse_last_cycle_raw": "ultimo ciclo, target non censurato",
}


def evaluate_one(record: dict, design) -> tuple[dict, np.ndarray, int, float]:
    """Riaddestra e valuta un modello, contando gli avvisi di mancata convergenza.

    Gli avvisi non vengono soppressi ma contati, con la stessa regola di lettura
    usata nei blocchi: sulla rete segnalano che l'ottimizzazione ha raggiunto il
    numero di iterazioni previsto, che è il meccanismo voluto; sui modelli a
    margine segnalano una stima troncata dal tetto alle iterazioni, cioè un
    punteggio che non è confrontabile con gli altri.
    """
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, y_pred = evaluate_holdout(
            record["estimator"],
            design.X_train,
            design.y_train,
            design.X_test,
            design.y_test,
            design.last_cycle,
            y_test_raw=design.y_test_raw,
        )
    seconds = time.perf_counter() - start
    n_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return result, y_pred, n_warnings, seconds


def check_reference(table: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Confronta le righe già note con i punteggi registrati alla convalida."""
    rows = []
    for (ref_subset, key), (all_cycles, last_cycle) in REFERENCE_HOLDOUT.items():
        if ref_subset != subset:
            continue
        found = table[table["model"] == key]
        if found.empty:
            continue
        found = found.iloc[0]
        for column, expected in (
            ("rmse_all_cycles", all_cycles),
            ("rmse_last_cycle", last_cycle),
        ):
            gap = abs(float(found[column]) - expected)
            rows.append(
                {
                    "subset": subset,
                    "model": key,
                    "lettura": READINGS[column],
                    "atteso": expected,
                    "ottenuto": float(found[column]),
                    "scarto": gap,
                    "coincide": gap <= REFERENCE_TOLERANCE,
                }
            )
    return pd.DataFrame(rows)


def rank_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Correlazione di rango fra la graduatoria in cross-validation e la verifica.

    Calcolata sui soli modelli, escluse le baseline: queste ultime sono ultime
    in ogni lettura e la loro presenza gonfierebbe la concordanza misurata senza
    dire nulla sull'ordine fra i modelli, che è la quantità di interesse.

    È una lettura descrittiva e fuori dal materiale del corso. Non è una
    statistica test: i punteggi della verifica sono singoli e privi di misura di
    variabilità, e la graduatoria del progetto resta quella in
    cross-validation.
    """
    models = table[~table["model"].isin(BASELINE_KEYS)]
    rows = []
    for column, name in READINGS.items():
        if column not in models:
            continue
        rho, _ = spearmanr(models["rmse_cv_mean"], models[column])
        rows.append(
            {
                "subset": models["subset"].iloc[0],
                "lettura": name,
                "spearman_con_cv": float(rho),
                "n_modelli": int(len(models)),
            }
        )
    return pd.DataFrame(rows)


def run_subset(subset: str, cap: int | None, only: list[str] | None) -> dict:
    design = build_design(subset, cap=cap)
    ranking = load_ranking(subset)
    if only:
        missing = sorted(set(only) - set(ranking["model"]))
        if missing:
            raise KeyError(f"{subset}: {missing} non sono nella graduatoria")
        ranking = ranking[ranking["model"].isin(only)]

    # La ricostruzione dell'intera graduatoria è anche la verifica che il
    # registro nella repository sia coerente con tutti gli artefatti prodotti.
    # Avviene prima di qualunque addestramento: se una configurazione non è
    # ricostruibile, il difetto va scoperto in un secondo e non a metà di una
    # corsa lunga.
    records = rebuild_all(design, ranking)

    print(f"\n=== {subset} ===")
    print(
        f"addestramento {len(design.X_train)} righe su {design.n_units_train} motori, "
        f"verifica {len(design.X_test)} righe su {design.n_units_test} motori, "
        f"{len(records)} modelli da leggere"
    )

    rows = []
    predictions = []
    for record in records:
        result, y_pred, n_warnings, seconds = evaluate_one(record, design)

        row = {
            "subset": subset,
            "blocco": record["blocco"],
            "label": record["label"],
            "model": record["model"],
            "config": record["config"],
            "rmse_cv_mean": record["rmse_cv_mean"],
            "rmse_cv_std": record["rmse_cv_std"],
        }
        row.update(result)
        row["convergence_warnings"] = n_warnings
        row["fit_seconds"] = seconds
        rows.append(row)

        predictions.append(
            pd.DataFrame(
                {
                    "model": record["model"],
                    "unit": design.groups_test,
                    "y_true": design.y_test,
                    "y_true_raw": design.y_test_raw,
                    "last_cycle": design.last_cycle,
                    "y_pred": y_pred.astype("float32"),
                }
            )
        )

        flag = f", {n_warnings} avvisi di convergenza" if n_warnings else ""
        print(
            f"[{record['model']}] cv {record['rmse_cv_mean']:.2f} | "
            f"tutti i cicli {result['rmse_all_cycles']:.2f} | "
            f"ultimo ciclo {result['rmse_last_cycle']:.2f} | "
            f"ultimo ciclo non censurato {result['rmse_last_cycle_raw']:.2f} "
            f"({seconds:.1f} s{flag})"
        )

    table = pd.DataFrame(rows)

    # L'ordine è quello della graduatoria in cross-validation e non viene
    # cambiato. Le colonne di rango rendono leggibile lo spostamento senza
    # riordinare la tabella, che equivarrebbe a promuovere l'insieme di verifica
    # a criterio di ordinamento.
    table["rango_cv"] = table["rmse_cv_mean"].rank(method="min").astype(int)
    for column, suffix in (
        ("rmse_all_cycles", "tutti"),
        ("rmse_last_cycle", "ultimo"),
        ("rmse_last_cycle_raw", "ultimo_grezzo"),
    ):
        table[f"rango_verifica_{suffix}"] = table[column].rank(method="min").astype(int)

    reference = check_reference(table, subset)
    agreement = rank_agreement(table)

    print("\ncontrollo di fedeltà sui punteggi registrati alla convalida")
    if reference.empty:
        print("nessuna riga di riferimento in questa esecuzione")
    else:
        print(reference.drop(columns=["subset"]).to_string(index=False))

    print("\nconcordanza di rango con la graduatoria in cross-validation")
    print(agreement.drop(columns=["subset"]).to_string(index=False))

    return {
        "holdout": table,
        "holdout_predictions": pd.concat(predictions, ignore_index=True),
        "holdout_agreement": agreement,
        "_reference": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="esecuzione parziale di convalida: stampa i risultati e non scrive artefatti",
    )
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for subset in args.subsets:
        outputs = run_subset(subset, args.cap, args.models)
        reference = outputs.pop("_reference")

        if args.models:
            print(f"\n{subset}: esecuzione parziale, artefatti non scritti")
        else:
            for name, frame in outputs.items():
                frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)
            print(f"\n{subset}: artefatti scritti in {OUTPUT_DIR}")

        if not reference.empty and not bool(reference["coincide"].all()):
            failures.append(reference[~reference["coincide"]])

    # Il controllo fallisce dopo la scrittura e non prima: se la catena dati è
    # cambiata, la tabella prodotta serve a capire dove e quanto, e ripetere
    # l'intera corsa per rileggerla sarebbe uno spreco.
    if failures:
        detail = pd.concat(failures, ignore_index=True).to_string(index=False)
        raise AssertionError(
            "i punteggi dei modelli deterministici non riproducono quelli "
            "registrati alla convalida del protocollo: la catena dati non è "
            f"quella che ha prodotto la graduatoria\n{detail}"
        )


if __name__ == "__main__":
    main()