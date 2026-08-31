"""Diagnostico: variabilita' dei modelli stocastici al variare del seme dello stimatore.

Ruolo nel progetto
    Il protocollo fissa un solo seme di stimatore per modello, uguale per tutti,
    e la dispersione riportata in graduatoria e' quella fra le 15 partizioni. Per
    i modelli con una componente casuale interna (inizializzazione dei pesi di
    una rete, campionamento delle righe e delle colonne di un insieme di alberi)
    quella dispersione non contiene la variabilita' dovuta al seme, che resta
    percio' non misurata.

    Il primo posto della graduatoria e' occupato su entrambi i sottoinsiemi da un
    modello di questo tipo, e i modelli che lo seguono a distanza inferiore alla
    soglia di leggibilita' sono anch'essi stocastici. Senza una misura di quella
    variabilita' non e' possibile dire se il primo posto sia una proprieta' del
    modello o dell'estrazione.

    Il risultato e' diagnostico e non entra in graduatoria, come il controllo con
    selezione annidata del blocco lineare. La graduatoria resta quella prodotta
    sotto il protocollo, con un seme per modello uguale per tutti: cambiare la
    regola per i soli modelli stocastici romperebbe la parita' del confronto.

Cosa riceve
    La graduatoria in `experiments/final/`, il registro dei modelli e i dati
    grezzi attraverso la catena `src.data`, `src.target`, `src.design`. Nessun
    argomento obbligatorio.

Cosa produce
    In `experiments/final/`, per ciascun sottoinsieme:

    - `{SUBSET}_seed_folds.csv`, le metriche di ogni modello su ogni partizione
      per ogni seme;
    - `{SUBSET}_seed_diagnostic.csv`, una riga per modello e seme con media e
      dispersione sulle 15 partizioni;
    - `{SUBSET}_seed_summary.csv`, una riga per modello con la dispersione delle
      medie fra semi, che e' la quantita' da confrontare con i divari della
      graduatoria.

    L'insieme di verifica ufficiale non viene letto.

Controllo incorporato
    Il primo seme dell'elenco e' quello del protocollo. Il punteggio ottenuto
    con quel seme deve coincidere con quello registrato in graduatoria: se non
    coincide, la ricostruzione del modello dagli artefatti non e' fedele e il
    resto della lettura non ha valore. Il confronto e' eseguito e riportato.

Come si lancia
    python -m scripts.run_seed_diagnostic
    python -m scripts.run_seed_diagnostic --subsets FD001 --models mlp
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.protocol import COMPARISON_SEEDS, N_SPLITS, check_no_group_leakage, evaluate, make_splits
from src.selected import load_ranking, rebuild
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final"

# Modelli sottoposti al diagnostico: quelli con una componente casuale interna
# che si trovano nel gruppo di testa della graduatoria su almeno un
# sottoinsieme. Gli altri modelli stocastici del confronto stanno oltre la
# soglia di leggibilita' dal primo posto, dove una variabilita' dell'ordine del
# decimo di ciclo non cambia la lettura.
STOCHASTIC_MODELS = ("mlp", "random_forest", "xgboost")

# Semi dello stimatore. Il primo e' quello del protocollo e serve da controllo:
# deve riprodurre il punteggio in graduatoria.
ESTIMATOR_SEEDS = (0, 1, 2, 3, 4)

# Tolleranza sulla riproduzione del punteggio in graduatoria con il seme del
# protocollo. Il valore assorbe le differenze di somma in virgola mobile fra
# esecuzioni, non una ricostruzione infedele, che si manifesterebbe
# sull'ordine del centesimo di ciclo o piu'.
REPRODUCTION_TOLERANCE = 1e-6


def with_seed(estimator, seed: int):
    """Copia dello stimatore con il seme della componente casuale sostituito."""
    if "model__random_state" not in estimator.get_params():
        raise AssertionError("lo stimatore non espone un seme sul passo del modello")
    return estimator.set_params(model__random_state=seed)


def run_subset(subset: str, models: list[str], cap: int | None) -> dict:
    design = build_design(subset, cap=cap)
    splits = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=COMPARISON_SEEDS)
    check_no_group_leakage(design.groups_train, splits)

    ranking = load_ranking(subset)
    print(f"\n=== {subset} ===")

    fold_frames = []
    records = []
    for key in models:
        row = ranking[ranking["model"] == key]
        if row.empty:
            print(f"[{key}] assente dalla graduatoria, saltato")
            continue
        row = row.iloc[0]
        base = rebuild(key, row["config"], design)
        print(f"\n[{key}] {row['config']}")

        for seed in ESTIMATOR_SEEDS:
            estimator = with_seed(base, seed)
            fold_metrics = evaluate(
                estimator, design.X_train, design.y_train, design.groups_train, splits
            )
            fold_metrics.insert(0, "model", key)
            fold_metrics.insert(1, "estimator_seed", seed)
            fold_frames.append(fold_metrics)

            mean = float(fold_metrics["rmse"].mean())
            std = float(fold_metrics["rmse"].std(ddof=1))
            records.append(
                {
                    "subset": subset,
                    "model": key,
                    "label": row["label"],
                    "estimator_seed": seed,
                    "rmse_mean": mean,
                    "rmse_std": std,
                    "fit_seconds_total": float(fold_metrics["fit_seconds"].sum()),
                }
            )
            print(f"    seme {seed}: rmse {mean:.4f} ± {std:.4f}")

            if seed == ESTIMATOR_SEEDS[0]:
                gap = abs(mean - float(row["rmse_mean"]))
                status = "coincide" if gap <= REPRODUCTION_TOLERANCE else "NON COINCIDE"
                print(
                    f"    controllo di riproduzione: {status} con la graduatoria "
                    f"({row['rmse_mean']:.6f}, scarto {gap:.2e})"
                )

    diagnostic = pd.DataFrame.from_records(records)
    summary = (
        diagnostic.groupby(["subset", "model", "label"], as_index=False)
        .agg(
            rmse_medio_fra_semi=("rmse_mean", "mean"),
            dispersione_fra_semi=("rmse_mean", "std"),
            rmse_minimo=("rmse_mean", "min"),
            rmse_massimo=("rmse_mean", "max"),
            n_semi=("estimator_seed", "count"),
        )
        .assign(escursione=lambda f: f["rmse_massimo"] - f["rmse_minimo"])
        .sort_values("rmse_medio_fra_semi")
    )

    print("\nvariabilita' della media sulle 15 partizioni al variare del seme")
    print(summary.drop(columns=["subset", "label"]).to_string(index=False))

    return {
        "seed_folds": pd.concat(fold_frames, ignore_index=True),
        "seed_diagnostic": diagnostic,
        "seed_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument("--models", nargs="+", default=list(STOCHASTIC_MODELS))
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        outputs = run_subset(subset, args.models, args.cap)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
