"""Esperimento del terzo blocco del confronto: la famiglia ad albero.

Ruolo nel progetto
    Compone l'esperimento del blocco lanciando, sotto il protocollo unico del
    progetto, i sei modelli dei laboratori 9 e 10 su ciascun sottoinsieme in
    perimetro. Non contiene logica di valutazione: quella sta in `src.protocol`
    e in `src.experiment`, e qui si decide soltanto quali modelli compongono il
    blocco, quali letture ne vengono estratte e dove ne vanno depositati gli
    artefatti.

    I laboratori 9 e 10 sono uniti in un blocco solo perche' la lettura centrale
    della famiglia e' albero singolo contro aggregazione contro addizione, cioe'
    riduzione della varianza contro riduzione della distorsione: distribuirla su
    due tabelle la trasformerebbe in un rimando fra artefatti.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`, `src.target`,
    `src.design`. Nessun argomento obbligatorio. I costi e la curva di
    saturazione su cui sono fissate le griglie sono misurati da
    `scripts/measure_tree_costs.py`, che precede questo script e non ne fa parte.

Cosa produce
    In `experiments/tree_models/`, per ciascun sottoinsieme:

    - `{SUBSET}_comparison.csv`, la tabella di confronto del blocco;
    - `{SUBSET}_cv_folds.csv`, le metriche di ogni modello su ognuna delle 15
      partizioni di confronto;
    - `{SUBSET}_grids.csv`, la griglia completa di ogni modello;
    - `{SUBSET}_importances.csv`, l'importanza per riduzione di impurita' di
      ciascuna variabile in ciascun modello;
    - `{SUBSET}_permutation_importances.csv`, l'importanza per permutazione,
      misurata sulle parti di verifica delle partizioni del seme di ricerca;
    - `{SUBSET}_diagnostics.csv`, configurazione selezionata, tempi, posizione
      rispetto ai bordi della griglia, configurazioni non valutabili e, per
      l'albero, la sua dimensione dopo la potatura;
    - `{SUBSET}_pruned_tree.joblib`, l'albero potato riaddestrato sull'intera
      parte di addestramento.

    L'insieme di verifica ufficiale non viene letto: la graduatoria si chiude
    quando tutti i blocchi del confronto sono conclusi, e solo allora i modelli
    selezionati vengono riaddestrati e valutati una volta sola.

L'albero serializzato
    La struttura dell'albero potato e' una delle letture richieste dal
    laboratorio 9 e va disegnata, non descritta. Il disegno e' compito del
    notebook, che pero' non addestra: l'albero viene percio' riaddestrato qui
    sull'intera parte di addestramento, come gia' avviene per l'estrazione dei
    parametri leggibili di ogni modello, e depositato accanto agli altri
    artefatti. Il file non e' versionato, come tutto cio' che sta in
    `experiments/`: la figura che ne deriva lo e'.

Colonna `n_nonzero` nella tabella di confronto
    Conta le variabili che il modello ha effettivamente usato per almeno una
    divisione, cioe' quelle con importanza non nulla. E' confrontabile lungo
    l'intera riga del blocco e con gli altri blocchi, dove la stessa colonna
    conta i coefficienti non annullati.

Come si lancia
    python -m scripts.run_tree_models
    python -m scripts.run_tree_models --subsets FD001 --quick
    python -m scripts.run_tree_models --models random_forest --n-jobs 4

    La modalita' `--quick` esegue la catena su griglie ridotte e serve a
    convalidarla prima di lanciare la versione completa.

    Il numero di processi paralleli va ridotto sulla foresta e sul bagging se la
    memoria e' scarsa: un insieme di 300 alberi non potati ha quasi quattro
    milioni di nodi e occupa circa 240 MB, e la ricerca su griglia ne tiene in
    vita una copia per processo.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.base import clone

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design, describe
from src.experiment import baseline_runs, comparison_table, run_grid_model
from src.protocol import (
    COMPARISON_SEEDS,
    N_SPLITS,
    SEARCH_SEEDS,
    check_no_group_leakage,
    make_splits,
)
from src.registry import TREE_MODELS
from src.target import RUL_CAP
from src.trees import permutation_importances, tree_summary

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "tree_models"

# Modello di cui viene serializzata la struttura. E' la lettura con cui il
# laboratorio 9 commenta la potatura, e non ha equivalente negli altri modelli
# del blocco, che sono insiemi di centinaia di alberi.
STRUCTURE_MODEL = "tree"

# Ripetizioni della permutazione di ciascuna variabile su ciascuna partizione.
# Cinque bastano a separare le variabili usate da quelle inerti; il costo
# cresce linearmente nel loro numero e nel numero di variabili.
PERMUTATION_REPEATS = 5


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


def run_subset(
    subset: str,
    cap: int | None,
    models: list[str],
    quick: bool,
    n_jobs: int,
    with_permutation: bool,
) -> dict:
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
    permutation_frames = []
    pruned_tree = None

    for key, spec in TREE_MODELS.items():
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

        # L'albero potato viene riaddestrato sull'intera parte di addestramento
        # e conservato. Il motore di esperimento compie gia' questo
        # riaddestramento per estrarre i parametri leggibili, ma non restituisce
        # il modello: ripeterlo qui costa un solo adattamento ed evita di
        # cambiare la firma del motore per il bisogno di un unico modello.
        if key == STRUCTURE_MODEL:
            pruned_tree = clone(run.estimator).fit(design.X_train, design.y_train)
            structure = tree_summary(pruned_tree)
            run.diagnostics.update(structure)
            print(
                f"    albero potato: {structure['n_leaves']} foglie, "
                f"profondita' {structure['depth']}"
            )

        if with_permutation:
            # Misurata sulle partizioni del seme di ricerca e non sulle quindici
            # di confronto: non e' una stima di prestazione e non entra in
            # graduatoria, quindi la ripetizione su tre semi ne triplicherebbe
            # il costo senza cambiarne la lettura.
            frame = permutation_importances(
                run.estimator,
                design,
                search_splits,
                n_repeats=2 if quick else PERMUTATION_REPEATS,
                seed=0,
            )
            frame.insert(0, "model", key)
            permutation_frames.append(frame)
            top = frame.iloc[0]
            print(
                f"    permutazione: variabile piu' rilevante {top['feature']}, "
                f"{top['importance_mean']:.2f} cicli di aumento dell'errore"
            )

        runs.append(run)

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

    importances = [r.coefficients for r in runs if r.coefficients is not None]
    if importances:
        outputs["importances"] = pd.concat(importances, ignore_index=True)

    if permutation_frames:
        outputs["permutation_importances"] = pd.concat(permutation_frames, ignore_index=True)

    diagnostics = pd.DataFrame(
        [{"model": r.key, "label": r.label, **r.diagnostics} for r in runs if r.diagnostics]
    )
    if not diagnostics.empty:
        outputs["diagnostics"] = diagnostics

    return {"tables": outputs, "pruned_tree": pruned_tree}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TREE_MODELS),
        help="sottoinsieme dei modelli del blocco da eseguire",
    )
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument("--no-cap", action="store_true", help="disattiva la censura del target")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="griglie ridotte: convalida della catena",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="salta l'importanza per permutazione, che e' la parte piu' lenta",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    cap = None if args.no_cap else args.cap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        result = run_subset(
            subset,
            cap,
            args.models,
            args.quick,
            args.n_jobs,
            with_permutation=not args.no_permutation,
        )
        for name, frame in result["tables"].items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)
        if result["pruned_tree"] is not None:
            joblib.dump(result["pruned_tree"], OUTPUT_DIR / f"{subset}_pruned_tree.joblib")

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
