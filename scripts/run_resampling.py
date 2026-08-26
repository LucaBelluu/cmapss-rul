"""Confronto fra i metodi di stima dell'errore del laboratorio 6.

Ruolo nel progetto
    Applica a un solo modello, la regressione lineare multipla, le quattro
    procedure di stima dell'errore viste nel corso, ricampionando le unita'
    motore anziche' le righe. Non e' un esperimento sul modello: e' un
    esperimento sulle procedure, e serve a mostrare quanto la stima
    dell'errore dipenda da come viene costruita.

    E' anche la giustificazione empirica dello schema adottato dal protocollo
    del progetto: il numero di fold e la ripetizione su piu' semi sono scelte
    che qui vengono misurate invece di essere soltanto argomentate.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`,
    `src.target`, `src.design`.

Cosa produce
    In `experiments/resampling/`, per ciascun sottoinsieme:

    - `{SUBSET}_methods.csv`, il riepilogo dei quattro metodi;
    - `{SUBSET}_validation_repeats.csv`, le venti ripetizioni della partizione
      unica, da cui si legge la dipendenza dalla partizione;
    - `{SUBSET}_leave_one_unit_out.csv`, l'errore su ciascuno dei cento motori;
    - `{SUBSET}_bootstrap_metrics.csv`, l'errore fuori campione dei campioni
      bootstrap;
    - `{SUBSET}_bootstrap_coefficients.csv`, la distribuzione dei coefficienti;
    - `{SUBSET}_coefficient_intervals.csv`, il riepilogo per variabile con
      l'indicazione di quali coefficienti mantengono il segno.

Come si lancia
    python -m scripts.run_resampling
    python -m scripts.run_resampling --subsets FD001 --bootstrap-samples 50
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.pipeline import build_pipeline
from src.resampling import (
    N_BOOTSTRAP,
    N_VALIDATION_REPEATS,
    bootstrap_estimates,
    coefficient_intervals,
    k_fold,
    leave_one_unit_out,
    summarize_methods,
    validation_set_approach,
)
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "resampling"


def run_subset(subset: str, cap: int | None, n_bootstrap: int, n_repeats: int) -> dict:
    design = build_design(subset, cap=cap)
    estimator = build_pipeline(LinearRegression())
    X, y, groups = design.X_train, design.y_train, design.groups_train
    print(f"\n=== {subset} ===")
    print(f"{len(X)} righe, {len(np.unique(groups))} motori, {len(design.features)} variabili")

    validation = validation_set_approach(estimator, X, y, groups, n_repeats=n_repeats)
    print(
        f"partizione unica: rmse {validation['rmse'].mean():.2f} ± {validation['rmse'].std(ddof=1):.2f} "
        f"su {len(validation)} ripetizioni, da {validation['rmse'].min():.2f} a {validation['rmse'].max():.2f}"
    )

    loo = leave_one_unit_out(estimator, X, y, groups)
    print(
        f"esclusione di un motore per volta: rmse {loo['rmse'].mean():.2f} ± {loo['rmse'].std(ddof=1):.2f} "
        f"su {len(loo)} stime, da {loo['rmse'].min():.2f} a {loo['rmse'].max():.2f}"
    )

    kf5 = k_fold(estimator, X, y, groups, n_splits=5)
    kf10 = k_fold(estimator, X, y, groups, n_splits=10)
    print(f"K-Fold a 5: rmse {kf5['rmse'].mean():.2f} ± {kf5['rmse'].std(ddof=1):.2f} su {len(kf5)} stime")
    print(f"K-Fold a 10: rmse {kf10['rmse'].mean():.2f} ± {kf10['rmse'].std(ddof=1):.2f} su {len(kf10)} stime")

    coefficients, boot_metrics = bootstrap_estimates(
        estimator, X, y, groups, design.features, n_samples=n_bootstrap
    )
    print(
        f"bootstrap: rmse {boot_metrics['rmse'].mean():.2f} ± {boot_metrics['rmse'].std(ddof=1):.2f} "
        f"su {len(boot_metrics)} campioni, in media {boot_metrics['n_valid_units'].mean():.1f} motori "
        f"esclusi su {len(np.unique(groups))}"
    )

    intervals = coefficient_intervals(coefficients)
    unstable = intervals.loc[~intervals["stable_sign"], "feature"].tolist()
    print(
        f"coefficienti con segno non stabile sui campioni bootstrap "
        f"({len(unstable)} su {len(intervals)}): {', '.join(unstable) if unstable else 'nessuno'}"
    )

    methods = summarize_methods([validation, loo, kf5, kf10, boot_metrics])
    methods.insert(0, "subset", subset)
    print("\nconfronto fra i metodi di stima dell'errore")
    print(methods.drop(columns=["subset"]).to_string(index=False))

    return {
        "methods": methods,
        "validation_repeats": validation,
        "leave_one_unit_out": loo,
        "bootstrap_metrics": boot_metrics,
        "bootstrap_coefficients": coefficients,
        "coefficient_intervals": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--validation-repeats", type=int, default=N_VALIDATION_REPEATS)
    args = parser.parse_args()

    cap = None if args.no_cap else args.cap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        outputs = run_subset(subset, cap, args.bootstrap_samples, args.validation_repeats)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()