"""Esperimento del secondo blocco del confronto: il superamento della linearita'.

Ruolo nel progetto
    Compone l'esperimento del blocco lanciando, sotto il protocollo unico del
    progetto, i quattro modelli del laboratorio 8 su ciascun sottoinsieme in
    perimetro. Non contiene logica di valutazione: quella sta in `src.protocol`
    e in `src.experiment`, e qui si decide soltanto quali modelli compongono il
    blocco e dove ne vanno depositati gli artefatti.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`,
    `src.target`, `src.design`. Nessun argomento obbligatorio.

Cosa produce
    In `experiments/nonlinear_models/`, per ciascun sottoinsieme:

    - `{SUBSET}_comparison.csv`, la tabella di confronto del blocco;
    - `{SUBSET}_cv_folds.csv`, le metriche di ogni modello su ognuna delle 15
      partizioni di confronto;
    - `{SUBSET}_grids.csv`, la griglia completa di ogni modello;
    - `{SUBSET}_terms.csv`, i termini della base espansa di ciascun modello con
      la variabile di provenienza, e per il modello additivo il riepilogo per
      variabile;
    - `{SUBSET}_partial_dependence.csv`, le funzioni parziali del modello
      additivo campionate su una griglia;
    - `{SUBSET}_diagnostics.csv`, configurazione selezionata, tempi, posizione
      rispetto ai bordi della griglia e mancate convergenze.

    L'insieme di verifica ufficiale non viene letto: la graduatoria si chiude
    quando tutti i blocchi del confronto sono conclusi, e solo allora i modelli
    selezionati vengono riaddestrati e valutati una volta sola.

Colonna `n_nonzero` nella tabella di confronto
    Conta le righe della tabella dei parametri leggibili, che per i tre modelli
    a espansione di base sono i termini della base e per il modello additivo
    sono le variabili. Fra i due gruppi la colonna non e' quindi confrontabile:
    un polinomio di grado 3 su 18 variabili ha 1.329 termini, un modello
    additivo ne ha 18 comunque penalizzati. La quantita' confrontabile per il
    modello additivo sono i gradi di liberta' effettivi, riportati in
    `{SUBSET}_terms.csv`.

Come si lancia
    python -m scripts.run_nonlinear_models
    python -m scripts.run_nonlinear_models --subsets FD001 --quick
    python -m scripts.run_nonlinear_models --models gam

    La modalita' `--quick` esegue la catena su griglie ridotte e serve a
    convalidarla prima di lanciare la versione completa.

    Il numero di processi paralleli va ridotto se la regola sui bordi imponesse
    di estendere la griglia del polinomio al grado 4: quella configurazione
    genera 7.314 colonne e la matrice espansa occupa centinaia di megabyte in
    ciascun processo.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.base import clone

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design, describe
from src.experiment import baseline_runs, comparison_table, run_grid_model
from src.nonlinear import gam_partial_dependence
from src.protocol import (
    COMPARISON_SEEDS,
    N_SPLITS,
    SEARCH_SEEDS,
    check_no_group_leakage,
    make_splits,
)
from src.registry import NONLINEAR_MODELS
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "nonlinear_models"

# Modello di cui vengono esportate le funzioni parziali. E' la lettura con cui
# il laboratorio commenta il modello additivo, e non ha equivalente negli altri
# modelli del blocco, i cui termini sono gia' in `{SUBSET}_terms.csv`.
PARTIAL_DEPENDENCE_MODEL = "gam"


def reduced_grid(grid: dict) -> dict:
    """Griglia ridotta agli estremi e al centro, per la modalita' di convalida.

    Esercita la catena su ogni parametro senza pagare il costo della griglia
    intera. I risultati prodotti in questa modalita' non entrano in nessuna
    tabella del progetto.
    """
    reduced = {}
    for name, values in grid.items():
        values = list(values)
        reduced[name] = values if len(values) <= 3 else [values[0], values[len(values) // 2], values[-1]]
    return reduced


def run_subset(subset: str, cap: int | None, models: list[str], quick: bool, n_jobs: int) -> dict:
    design = build_design(subset, cap=cap)
    print(f"\n=== {subset} ===")
    print(describe(design).to_string())
    print(f"variabili ({len(design.features)}): {', '.join(design.features)}")

    search_splits = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=SEARCH_SEEDS)
    comparison_splits = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=COMPARISON_SEEDS)
    check_no_group_leakage(design.groups_train, search_splits)
    check_no_group_leakage(design.groups_train, comparison_splits)
    print(
        f"ricerca su {len(search_splits)} partizioni, confronto su {len(comparison_splits)}, "
        f"nessuna sovrapposizione fra motori"
    )

    runs = baseline_runs(design, comparison_splits)
    partial_dependence = None

    for key, spec in NONLINEAR_MODELS.items():
        if key not in models:
            continue
        print(f"\n[{key}] ricerca e rivalutazione")

        spec_used = spec
        if quick and spec.grid and not callable(spec.grid):
            spec_used = type(spec)(
                key=spec.key,
                label=spec.label,
                estimator=spec.estimator,
                grid=reduced_grid(spec.grid),
                reader=spec.reader,
                note=spec.note + " (griglia ridotta, modalita' di convalida)",
            )

        run = run_grid_model(
            spec_used,
            design,
            search_splits=search_splits,
            comparison_splits=comparison_splits,
            n_jobs=n_jobs,
        )
        print(
            f"    configurazione: {run.summary['config']} | "
            f"rmse {run.summary['rmse_mean']:.2f} ± {run.summary['rmse_std']:.2f} | "
            f"ricerca {run.summary['search_seconds']:.0f} s su "
            f"{int(run.summary['n_configurations'])} configurazioni"
        )
        if run.diagnostics.get("boundary"):
            print(
                f"    ATTENZIONE: configurazione sul bordo della griglia "
                f"({run.diagnostics['boundary']})"
            )
        if run.diagnostics.get("convergence_warnings"):
            print(f"    ATTENZIONE: {run.diagnostics['convergence_warnings']} mancate convergenze")
        if run.diagnostics.get("failed_configurations"):
            print(
                f"    ATTENZIONE: {run.diagnostics['failed_configurations']} configurazioni "
                f"non valutabili, escluse dalla ricerca senza essere provate"
            )

        # Numero di colonne effettivamente generate dall'espansione, contro
        # quelle che la configurazione selezionata produrrebbe se ogni variabile
        # le ricevesse tutte. La differenza sono gli intervalli degeneri rimossi
        # dalla discretizzazione su variabili a pochi valori distinti: e' la
        # quantita' da cui si legge quel fenomeno, che nel registro
        # dell'esecuzione non compare piu' come avviso ripetuto.
        if key == "step_functions" and run.coefficients is not None:
            produced = len(run.coefficients)
            expected = len(design.features) * int(run.config["model__expand__n_bins"])
            run.diagnostics["expansion_columns"] = produced
            run.diagnostics["expansion_columns_nominal"] = expected
            print(f"    colonne generate: {produced} su {expected} nominali")

        runs.append(run)

        # Le funzioni parziali richiedono il modello adattato sull'intera parte
        # di addestramento, che il motore di esperimento usa per estrarre i
        # parametri leggibili senza restituirlo. Riaddestrarlo qui costa un
        # solo adattamento ed evita di cambiare la firma del motore per un
        # bisogno di un unico modello.
        if key == PARTIAL_DEPENDENCE_MODEL:
            fitted = clone(run.estimator).fit(design.X_train, design.y_train)
            partial_dependence = gam_partial_dependence(fitted, list(design.features))
            partial_dependence.insert(0, "model", key)

    table = comparison_table(runs)
    print("\ntabella di confronto del blocco")
    print(table.drop(columns=["subset"]).to_string(index=False))

    outputs = {
        "comparison": table,
        "cv_folds": pd.concat([r.fold_metrics for r in runs], ignore_index=True),
    }

    grids = [r.grid_table.assign(model=r.key) for r in runs if r.grid_table is not None]
    if grids:
        outputs["grids"] = pd.concat(grids, ignore_index=True)

    terms = [r.coefficients for r in runs if r.coefficients is not None]
    if terms:
        outputs["terms"] = pd.concat(terms, ignore_index=True)

    if partial_dependence is not None:
        outputs["partial_dependence"] = partial_dependence

    diagnostics = pd.DataFrame(
        [{"model": r.key, "label": r.label, **r.diagnostics} for r in runs if r.diagnostics]
    )
    if not diagnostics.empty:
        outputs["diagnostics"] = diagnostics

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(NONLINEAR_MODELS),
        help="sottoinsieme dei modelli del blocco da eseguire",
    )
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument("--no-cap", action="store_true", help="disattiva la censura del target")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="griglie ridotte: convalida della catena",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    cap = None if args.no_cap else args.cap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        outputs = run_subset(subset, cap, args.models, args.quick, args.n_jobs)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()