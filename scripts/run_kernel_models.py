"""Esperimento del quarto blocco del confronto: metodi a margine e reti.

Ruolo nel progetto
    Compone l'esperimento del blocco lanciando, sotto il protocollo unico del
    progetto, i quattro modelli del laboratorio 11 su ciascun sottoinsieme in
    perimetro. Non contiene logica di valutazione: quella sta in `src.protocol`
    e in `src.experiment`, e qui si decide soltanto quali modelli compongono il
    blocco, quali letture ne vengono estratte e dove ne vanno depositati gli
    artefatti.

    Le tre varianti di kernel e la rete stanno in un blocco solo perche' la
    lettura centrale e' la stessa per entrambe le famiglie: cosa guadagna una
    funzione non lineare stimata senza espansione esplicita delle variabili
    rispetto ai modelli che quella espansione la costruiscono. Separarle
    produrrebbe due tabelle da leggere l'una accanto all'altra.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`, `src.target`,
    `src.design`. Nessun argomento obbligatorio. I costi su cui sono fissate le
    griglie sono misurati da `scripts/measure_kernel_costs.py`, che precede
    questo script e non ne fa parte.

Cosa produce
    In `experiments/kernel_models/`, accanto agli artefatti della sonda dei
    costi, per ciascun sottoinsieme:

    - `{SUBSET}_comparison.csv`, la tabella di confronto del blocco;
    - `{SUBSET}_cv_folds.csv`, le metriche di ogni modello su ognuna delle 15
      partizioni di confronto;
    - `{SUBSET}_grids.csv`, la griglia completa di ogni modello;
    - `{SUBSET}_coefficients.csv`, i coefficienti della variante a kernel
      lineare, che e' l'unico modello del blocco a esporli;
    - `{SUBSET}_permutation_importances.csv`, l'importanza per permutazione di
      ciascuna variabile in ciascun modello, misurata sulle parti di verifica
      delle partizioni del seme di ricerca;
    - `{SUBSET}_diagnostics.csv`, configurazione selezionata, tempi, posizione
      rispetto ai bordi della griglia, configurazioni non valutabili e, per
      ciascun modello, il riepilogo strutturale della soluzione.

    L'insieme di verifica ufficiale non viene letto: la graduatoria si chiude
    quando tutti i blocchi del confronto sono conclusi, e solo allora i modelli
    selezionati vengono riaddestrati e valutati una volta sola.

Perche' l'importanza per permutazione anche qui
    Tre modelli su quattro non espongono coefficienti, quindi senza questa
    misura la loro riga della tabella resterebbe senza alcuna lettura sulle
    variabili, e il blocco non sarebbe commentabile alla pari degli altri. La
    misura e' quella del blocco ad albero, sulle stesse partizioni e con lo
    stesso numero di ripetizioni, quindi i valori sono confrontabili fra i due
    blocchi.

    E' anche la parte piu' lenta dell'esecuzione, perche' richiede una
    predizione per ogni variabile e per ogni ripetizione, e la predizione di un
    modello a margine costa quanto il calcolo del kernel contro tutti i suoi
    vettori di supporto. L'opzione `--no-permutation` la salta.

Colonna `n_nonzero` nella tabella di confronto
    E' definita per la sola variante a kernel lineare, dove conta i coefficienti
    non nulli come negli altri blocchi. Per gli altri tre modelli resta vuota:
    la funzione stimata non ha coefficienti sulle variabili, e riempire la
    colonna con il numero di vettori di supporto vi metterebbe una quantita'
    diversa sotto la stessa intestazione. Il numero di vettori di supporto e' in
    `{SUBSET}_diagnostics.csv`.

Come si lancia
    python -m scripts.run_kernel_models
    python -m scripts.run_kernel_models --subsets FD001 --quick
    python -m scripts.run_kernel_models --models svr_rbf --no-permutation

    La modalita' `--quick` esegue la catena su griglie ridotte e serve a
    convalidarla prima di lanciare la versione completa.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.base import clone

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design, describe
from src.experiment import baseline_runs, comparison_table, run_grid_model
from src.margin import network_summary, support_summary
from src.protocol import (
    COMPARISON_SEEDS,
    N_SPLITS,
    SEARCH_SEEDS,
    check_no_group_leakage,
    make_splits,
)
from src.registry import KERNEL_MODELS
from src.target import RUL_CAP
from src.trees import permutation_importances

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "kernel_models"

# Modello di cui vengono estratti i coefficienti. E' l'unico del blocco la cui
# funzione stimata resta lineare nelle variabili.
COEFFICIENT_MODEL = "svr_linear"

# Modello la cui lettura strutturale riguarda una rete e non una soluzione a
# margine.
NETWORK_MODEL = "mlp"

# Ripetizioni della permutazione di ciascuna variabile su ciascuna partizione.
# E' lo stesso numero usato dal blocco ad albero: un numero diverso renderebbe
# le due tabelle non confrontabili.
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
        reduced[name] = (
            values if len(values) <= 3 else [values[0], values[len(values) // 2], values[-1]]
        )
    return reduced


def structural_summary(key: str, estimator, design) -> dict:
    """Riepilogo strutturale del modello selezionato, riaddestrato per intero.

    Il riaddestramento sull'intera parte di addestramento e' lo stesso che il
    motore di esperimento compie per estrarre i parametri leggibili, ma il
    motore non restituisce il modello adattato: ripeterlo qui costa un solo
    adattamento ed evita di cambiare la firma del motore per il bisogno di un
    unico blocco.
    """
    fitted = clone(estimator).fit(design.X_train, design.y_train)
    if key == NETWORK_MODEL:
        return network_summary(fitted)
    summary = support_summary(fitted)
    summary["support_share"] = summary["n_support"] / len(design.X_train)
    return summary


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

    for key, spec in KERNEL_MODELS.items():
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
        if run.diagnostics.get("failed_configurations"):
            print(
                f"    ATTENZIONE: {run.diagnostics['failed_configurations']} configurazioni "
                f"non valutabili, escluse dalla ricerca senza essere provate"
            )
        if run.diagnostics.get("convergence_warnings"):
            # Sui modelli a margine l'avviso segnala una stima troncata dal tetto
            # alle iterazioni, quindi un punteggio non confrontabile. Sulla rete
            # segnala l'arresto al numero di iterazioni previsto, che e' il
            # meccanismo voluto: la distinzione e' nel messaggio, non nel conteggio.
            kind = (
                "arresti al numero di iterazioni previsto"
                if key == NETWORK_MODEL
                else "stime troncate dal tetto alle iterazioni"
            )
            print(f"    {run.diagnostics['convergence_warnings']} {kind}")

        structure = structural_summary(key, run.estimator, design)
        run.diagnostics.update(structure)
        if key == NETWORK_MODEL:
            print(
                f"    rete: {structure['n_parameters']} parametri, "
                f"{structure['n_iter']} iterazioni, perdita finale "
                f"{structure['final_loss']:.1f}"
            )
        else:
            print(
                f"    soluzione: {structure['n_support']:,} vettori di supporto "
                f"({structure['support_share']:.1%} delle righe)"
            )

        if with_permutation:
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

    coefficients = [r.coefficients for r in runs if r.coefficients is not None]
    if coefficients:
        outputs["coefficients"] = pd.concat(coefficients, ignore_index=True)

    if permutation_frames:
        outputs["permutation_importances"] = pd.concat(permutation_frames, ignore_index=True)

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
        default=list(KERNEL_MODELS),
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
        outputs = run_subset(
            subset,
            cap,
            args.models,
            args.quick,
            args.n_jobs,
            with_permutation=not args.no_permutation,
        )
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
