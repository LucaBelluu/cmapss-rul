"""Convalida della catena di valutazione end to end.

Ruolo nel progetto
    Esercita l'intero protocollo su modelli deliberatamente semplici, prima che
    venga applicato ai modelli del confronto. Non produce risultati destinati
    alla tabella finale: produce le prove che il protocollo si comporta come
    dichiarato.

Cosa riceve
    I file grezzi in `data/raw/`, tramite `src.data`. Nessun argomento
    obbligatorio; i sottoinsiemi e la soglia di censura sono parametrizzabili.

Cosa produce
    In `experiments/protocol_check/`, per ciascun sottoinsieme: il riepilogo
    della matrice di progetto, le metriche per fold, il riepilogo per modello,
    il confronto fra partizionamento per unita' e partizionamento per riga, e la
    valutazione sull'insieme di verifica ufficiale.

Cosa verifica
    1. Che il target di verifica sull'ultimo ciclo di ogni unita' coincida con
       le etichette del file di RUL. Il controllo e' dentro `src.design` e fa
       fallire la costruzione della matrice.
    2. Che nessun motore compaia contemporaneamente in addestramento e in
       verifica in nessuna partizione.
    3. Che la baseline costante produca un errore quadratico medio pari alla
       deviazione standard del target, il che conferma che target e metriche
       sono allineati.
    4. Che il partizionamento per riga produca un errore inferiore a quello per
       unita', cioe' che il vincolo di gruppo stia effettivamente correggendo
       una stima ottimistica. Il confronto e' eseguito su un modello lineare e
       su un insieme di alberi: il primo ha capacita' limitata di sfruttare la
       somiglianza fra cicli adiacenti, il secondo no, e il divario fra i due
       divari misura quanto il vincolo conti.
    5. Che il divario fra errore in cross-validation ed errore sull'insieme di
       verifica ufficiale sia leggibile. Le due misure riguardano popolazioni di
       cicli diverse e non si sottraggono: la lettura si fa sull'R quadro.

Modelli impiegati
    Nessuno di essi appartiene al confronto. Le due baseline sono i termini di
    paragone della tabella finale, la regressione lineare e' il modello con cui
    si esercita la catena, la regressione senza numero di ciclo e la foresta
    casuale sono strumenti diagnostici.

Come si lancia
    python -m scripts.run_protocol_check
    python -m scripts.run_protocol_check --subsets FD001 --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from src.baselines import all_baselines
from src.data import CYCLE_COL, PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design, describe
from src.pipeline import build_pipeline
from src.protocol import (
    COMPARISON_SEEDS,
    FoldSplit,
    N_SPLITS,
    check_no_group_leakage,
    evaluate,
    evaluate_holdout,
    make_splits,
    summarize,
)
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "protocol_check"

# Seme fisso per la foresta usata nella diagnosi del partizionamento. Non e' un
# modello del confronto: serve solo a rendere visibile l'effetto del vincolo di
# gruppo su un modello capace di memorizzare le righe vicine.
DIAGNOSTIC_SEED = 0


def row_wise_splits(n_rows: int, n_splits: int, seeds) -> list[FoldSplit]:
    """Partizioni casuali per riga, senza vincolo di gruppo.

    Servono unicamente al confronto diagnostico: non vengono mai usate per
    produrre risultati del progetto.
    """
    splits: list[FoldSplit] = []
    placeholder = np.zeros(n_rows)
    for seed in seeds:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(placeholder)):
            splits.append(FoldSplit(seed=seed, fold=fold, train=train_idx, valid=valid_idx))
    return splits


def run_subset(subset: str, cap: int | None, seeds, quick: bool) -> dict[str, pd.DataFrame]:
    design = build_design(subset, cap=cap)
    summary_design = describe(design)
    print(f"\n=== {subset} ===")
    print(summary_design.to_string())
    print(f"variabili ({len(design.features)}): {', '.join(design.features)}")
    print(f"rimosse ({len(design.dropped)}): {', '.join(design.dropped)}")
    print("target di verifica sull'ultimo ciclo coerente con le etichette RUL")

    splits = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=seeds)
    check_no_group_leakage(design.groups_train, splits)
    print(
        f"partizioni: {len(splits)} ({N_SPLITS} fold x {len(seeds)} semi), "
        f"nessuna sovrapposizione"
    )

    models = all_baselines()
    models["regressione_lineare"] = build_pipeline(LinearRegression())

    # Modello diagnostico, non in confronto: quantifica in cross-validation la
    # quota di capacita' predittiva che proviene dal solo conteggio dei cicli.
    # La relazione fra numero di ciclo e vita residua e' esatta sulle
    # traiettorie complete e non lo e' su quelle troncate, quindi questa quota
    # e' anche parte della spiegazione del divario fra cross-validation e
    # verifica finale.
    models["regressione_lineare_senza_ciclo"] = build_pipeline(
        LinearRegression(), columns=[c for c in design.features if c != CYCLE_COL]
    )

    fold_frames = []
    summaries = []
    for name, model in models.items():
        fold_metrics = evaluate(model, design.X_train, design.y_train, design.groups_train, splits)
        fold_metrics.insert(0, "model", name)
        fold_frames.append(fold_metrics)
        summaries.append(summarize(fold_metrics, label=name))

    cv_folds = pd.concat(fold_frames, ignore_index=True)
    cv_summary = pd.DataFrame(summaries)
    print("\ncross-validation per unita' motore")
    print(cv_summary.to_string(index=False))

    # Controllo di coerenza: l'errore quadratico medio della predizione costante
    # deve coincidere con la deviazione standard del target. Lo scarto residuo e'
    # dovuto al fatto che la costante e' la media dei motori di addestramento del
    # fold e non quella del fold di verifica.
    const_rmse = cv_summary.loc[cv_summary["model"] == "baseline_costante", "rmse_mean"].iloc[0]
    print(
        f"\ncoerenza baseline costante: rmse {const_rmse:.3f} "
        f"contro deviazione standard del target {design.y_train.std(ddof=1):.3f}"
    )

    # Diagnosi del partizionamento: stesso modello, stesso numero di fold, unica
    # differenza il vincolo di gruppo.
    diagnostic_seeds = seeds[:1]
    grouped = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=diagnostic_seeds)
    by_row = row_wise_splits(len(design.X_train), N_SPLITS, diagnostic_seeds)

    diagnostic_models = {"regressione_lineare": build_pipeline(LinearRegression())}
    if not quick:
        diagnostic_models["foresta_casuale"] = build_pipeline(
            RandomForestRegressor(
                n_estimators=100, random_state=DIAGNOSTIC_SEED, n_jobs=-1
            )
        )

    rows = []
    for name, model in diagnostic_models.items():
        for scheme, scheme_splits in (("per_unita", grouped), ("per_riga", by_row)):
            metrics = evaluate(
                model, design.X_train, design.y_train, design.groups_train, scheme_splits
            )
            rows.append(
                {
                    "model": name,
                    "scheme": scheme,
                    "rmse_mean": metrics["rmse"].mean(),
                    "rmse_std": metrics["rmse"].std(ddof=1),
                }
            )
    leakage = pd.DataFrame(rows)
    pivot = leakage.pivot(index="model", columns="scheme", values="rmse_mean")
    pivot["ottimismo"] = pivot["per_unita"] - pivot["per_riga"]
    pivot["ottimismo_relativo"] = pivot["ottimismo"] / pivot["per_unita"]
    print("\neffetto del vincolo di gruppo (rmse medio)")
    print(pivot.to_string())

    # Verifica finale, letta una sola volta. In questo script non chiude nessuna
    # graduatoria: serve a controllare che la catena arrivi in fondo e che i
    # numeri siano leggibili.
    holdout_rows = []
    for name, model in models.items():
        result, _ = evaluate_holdout(
            model,
            design.X_train,
            design.y_train,
            design.X_test,
            design.y_test,
            design.last_cycle,
            design.y_test_raw,
        )
        result["model"] = name
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    holdout = holdout[["model"] + [c for c in holdout.columns if c != "model"]]
    print("\ninsieme di verifica ufficiale")
    print(holdout.to_string(index=False))

    return {
        "design": summary_design.to_frame().T,
        "cv_folds": cv_folds,
        "cv_summary": cv_summary,
        "partitioning": leakage,
        "holdout": holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="disattiva la censura del target (controllo di sensibilita')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="un solo seme e nessuna foresta nella diagnosi del partizionamento",
    )
    args = parser.parse_args()

    cap = None if args.no_cap else args.cap
    seeds = COMPARISON_SEEDS[:1] if args.quick else COMPARISON_SEEDS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        outputs = run_subset(subset, cap, seeds, args.quick)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()