"""Esperimento del primo blocco del confronto: i modelli lineari del programma.

Ruolo nel progetto
    Compone l'esperimento del blocco lanciando, sotto il protocollo unico del
    progetto, i modelli lineari del laboratorio 7 su ciascun sottoinsieme in
    perimetro. Non contiene logica di valutazione: quella sta in `src.protocol`
    e in `src.experiment`, e qui si decide soltanto quali modelli compongono il
    blocco e dove ne vanno depositati gli artefatti.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`,
    `src.target`, `src.design`. Nessun argomento obbligatorio.

Cosa produce
    In `experiments/linear_models/`, per ciascun sottoinsieme:

    - `{SUBSET}_comparison.csv`, la tabella di confronto del blocco;
    - `{SUBSET}_cv_folds.csv`, le metriche di ogni modello su ognuna delle 15
      partizioni di confronto;
    - `{SUBSET}_grids.csv`, la griglia completa di ogni modello con
      iperparametri;
    - `{SUBSET}_coefficients.csv`, i parametri leggibili di ogni modello;
    - `{SUBSET}_selection_history.csv`, il percorso dei tre metodi di selezione
      delle variabili;
    - `{SUBSET}_coefficient_paths.csv`, i percorsi dei coefficienti di Ridge e
      Lasso al variare della penalizzazione;
    - `{SUBSET}_nested_check.csv`, la misura dell'ottimismo introdotto dalla
      selezione delle variabili non annidata;
    - `{SUBSET}_diagnostics.csv`, configurazione selezionata, tempi, posizione
      rispetto ai bordi della griglia e mancate convergenze.

    L'insieme di verifica ufficiale non viene letto: la graduatoria si chiude
    quando tutti i blocchi del confronto sono conclusi, e solo allora i modelli
    selezionati vengono riaddestrati e valutati una volta sola.

Come si lancia
    python -m scripts.run_linear_models
    python -m scripts.run_linear_models --subsets FD001 --quick
    python -m scripts.run_linear_models --models ridge lasso

    La modalita' `--quick` esegue la catena su una griglia ridotta e senza
    ricerca esaustiva, e serve a convalidarla prima di lanciare la versione
    completa.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.linear_model import Lasso, Ridge

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design, describe
from src.experiment import (
    baseline_runs,
    comparison_table,
    coefficient_path,
    nested_selection_check,
    run_grid_model,
    run_selection_model,
)
from src.protocol import (
    COMPARISON_SEEDS,
    N_SPLITS,
    SEARCH_SEEDS,
    check_no_group_leakage,
    make_splits,
)
from src.registry import (
    LASSO_ALPHAS,
    LINEAR_MODELS,
    RIDGE_ALPHAS,
    SELECTION_MODELS,
)
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "linear_models"

# Metodo di selezione su cui viene misurato l'ottimismo della selezione non
# annidata. La forward e' scelta perche' e' l'unico dei tre il cui costo
# annidato resta trascurabile: la misura vale come cautela di lettura per tutti
# e tre, che condividono lo stesso motore di stima e lo stesso criterio.
NESTED_CHECK_METHOD = "forward_stepwise"


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

    for key, spec in LINEAR_MODELS.items():
        if key not in models:
            continue
        print(f"\n[{key}] ricerca e rivalutazione")
        spec_used = spec
        if quick and callable(spec.grid) is False and spec.grid:
            # In modalita' di convalida la griglia e' ridotta ai suoi estremi e
            # al centro: esercita la catena senza pagarne il costo.
            reduced = {
                name: list(values)[:: max(1, len(list(values)) // 3)]
                for name, values in spec.grid.items()
            }
            spec_used = type(spec)(
                key=spec.key,
                label=spec.label,
                estimator=spec.estimator,
                grid=reduced,
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
            f"rmse {run.summary['rmse_mean']:.2f} ± {run.summary['rmse_std']:.2f}"
        )
        if run.diagnostics.get("boundary"):
            print(f"    ATTENZIONE: configurazione sul bordo della griglia ({run.diagnostics['boundary']})")
        if run.diagnostics.get("convergence_warnings"):
            print(f"    ATTENZIONE: {run.diagnostics['convergence_warnings']} mancate convergenze")
        runs.append(run)

    for method, label in SELECTION_MODELS.items():
        if method not in models:
            continue
        if quick and method == "best_subset":
            print("\n[best_subset] saltata in modalita' di convalida")
            continue
        print(f"\n[{method}] ricerca del sottoinsieme e rivalutazione")
        run = run_selection_model(
            method,
            label,
            design,
            search_splits=search_splits,
            comparison_splits=comparison_splits,
        )
        print(
            f"    k={run.config['k']} in {run.summary['search_seconds']:.1f} s | "
            f"rmse {run.summary['rmse_mean']:.2f} ± {run.summary['rmse_std']:.2f}"
        )
        print(f"    variabili: {', '.join(run.config['features'])}")
        runs.append(run)

    table = comparison_table(runs)
    print("\ntabella di confronto del blocco")
    print(table.drop(columns=["subset"]).to_string(index=False))

    outputs = {
        "comparison": table,
        "cv_folds": pd.concat([r.fold_metrics for r in runs], ignore_index=True),
    }

    grids = [
        r.grid_table.assign(model=r.key) for r in runs if r.grid_table is not None
    ]
    if grids:
        outputs["grids"] = pd.concat(grids, ignore_index=True)

    coefficients = [r.coefficients for r in runs if r.coefficients is not None]
    if coefficients:
        outputs["coefficients"] = pd.concat(coefficients, ignore_index=True)

    histories = [r.selection_history for r in runs if r.selection_history is not None]
    if histories:
        outputs["selection_history"] = pd.concat(histories, ignore_index=True)

    diagnostics = pd.DataFrame(
        [{"model": r.key, "label": r.label, **r.diagnostics} for r in runs if r.diagnostics]
    )
    if not diagnostics.empty:
        outputs["diagnostics"] = diagnostics

    # Percorsi dei coefficienti: descrivono come la penalizzazione spegne le
    # variabili, ed e' la lettura con cui il laboratorio commenta Ridge e Lasso.
    if not quick:
        paths = []
        for label, factory, values in (
            ("ridge", lambda **kw: Ridge(**kw), RIDGE_ALPHAS),
            ("lasso", lambda **kw: Lasso(max_iter=50_000, **kw), LASSO_ALPHAS),
        ):
            path = coefficient_path(
                factory, values, design.X_train, design.y_train, design.features
            )
            path.insert(0, "model", label)
            paths.append(path)
        outputs["coefficient_paths"] = pd.concat(paths, ignore_index=True)

        if NESTED_CHECK_METHOD in models:
            print(f"\n[{NESTED_CHECK_METHOD}] controllo diagnostico con selezione annidata")
            nested = nested_selection_check(
                NESTED_CHECK_METHOD, design, comparison_splits=comparison_splits
            )
            reported = table.loc[table["model"] == NESTED_CHECK_METHOD, "rmse_mean"]
            print(
                f"    rmse con selezione annidata {nested['rmse_outer'].mean():.2f} "
                f"± {nested['rmse_outer'].std(ddof=1):.2f} contro "
                f"{reported.iloc[0]:.2f} riportato in tabella"
            )
            print(
                f"    cardinalita' selezionate nei fold: "
                f"{sorted(nested['k'].unique().tolist())}"
            )
            outputs["nested_check"] = nested

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(LINEAR_MODELS) + list(SELECTION_MODELS),
        help="sottoinsieme dei modelli del blocco da eseguire",
    )
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument("--no-cap", action="store_true", help="disattiva la censura del target")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="griglie ridotte e nessuna ricerca esaustiva: convalida della catena",
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