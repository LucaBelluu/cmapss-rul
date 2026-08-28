"""Misura dei costi e della saturazione degli insiemi, preliminare al blocco della
famiglia ad albero.

Ruolo nel progetto
    Precede l'esperimento del terzo blocco del confronto e non ne fa parte. Le
    griglie degli iperparametri sono fissate su una misura del costo e non su
    una stima, come per il blocco precedente, e il numero di alberi degli
    insiemi per aggregazione e' fissato a priori con una curva che ne mostra la
    saturazione. Nessun risultato prodotto qui entra in graduatoria.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`, `src.target`,
    `src.design`. Nessun argomento obbligatorio.

Cosa produce
    In `experiments/tree_models/`, per ciascun sottoinsieme:

    - `{SUBSET}_cost_probe.csv`, tempo di adattamento e di predizione, numero di
      nodi dell'insieme e errore su una sola partizione, per gli angoli costosi
      delle griglie candidate;
    - `{SUBSET}_ensemble_saturation.csv`, errore e tempo cumulato al crescere del
      numero di alberi, per il bagging e per due configurazioni della foresta.

Perche' una sola partizione
    Il costo di un adattamento non richiede una media su piu' fold: dipende
    dalla forma della matrice, che e' la stessa su tutte le partizioni. La
    misura usa quindi la prima partizione del seme di ricerca, cioe' 80 motori
    in addestramento e 20 in verifica, che e' esattamente la forma su cui il
    blocco lavorera'.

Le due quantita' che contano
    Il tempo per adattamento moltiplicato per il numero di configurazioni e per
    il numero di fold da' la durata della ricerca di ciascun modello. Il numero
    di nodi dell'insieme e' un indice diretto della memoria occupata: un albero
    non potato su 16.500 righe ha circa una foglia per riga, e la ricerca su
    griglia ne tiene in vita tante copie quanti sono i processi paralleli. E' il
    vincolo operativo del blocco.

Errore riportato nella sonda dei costi
    L'errore sulla partizione e' registrato come controllo di plausibilita'
    della catena, non come criterio con cui fissare gli estremi delle griglie:
    scegliere un intervallo perche' contiene il valore migliore osservato in
    questa misura sarebbe una selezione fatta prima e fuori dal protocollo.

Curva di saturazione
    L'errore di un insieme per aggregazione decresce in modo monotono nel numero
    di alberi e satura: il numero di alberi non e' un iperparametro che governa
    un compromesso, ma un parametro di precisione della media. Metterlo in
    griglia farebbe selezionare sempre il valore massimo e chiederebbe alla
    regola sui bordi un'estensione senza fine. Il valore e' percio' fissato a
    300 e la curva serve a verificare che a quel punto non resti guadagno
    apprezzabile; se la curva fosse ancora in discesa, il valore sale prima che
    il blocco venga eseguito.

    La sequenza e' costruita con l'aggiunta incrementale di alberi a uno stesso
    insieme, quindi l'intera curva costa quanto il solo adattamento con il
    numero massimo di alberi.

Come si lancia
    python -m scripts.measure_tree_costs
    python -m scripts.measure_tree_costs --subsets FD001 --max-trees 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.pipeline import build_pipeline
from src.protocol import (
    N_SPLITS,
    SEARCH_SEEDS,
    make_splits,
    regression_metrics,
)
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "tree_models"

# Seme degli stimatori che ne richiedono uno. E' distinto dai semi del
# protocollo, che governano il partizionamento: qui riguarda il campionamento
# bootstrap e la scelta delle variabili candidate, non quali motori finiscono
# da che parte.
MODEL_SEED = 0

# Punti in cui la curva di saturazione viene letta.
SATURATION_CHECKPOINTS = (25, 50, 100, 200, 300, 400, 500)


def _n_nodes(estimator) -> float:
    """Nodi complessivi di un albero o di un insieme di alberi di scikit-learn.

    E' la quantita' da cui si legge la memoria occupata. Non e' disponibile
    sull'implementazione esterna di gradient boosting, che non espone gli alberi
    come oggetti di scikit-learn: per quel modello la colonna resta vuota.
    """
    if hasattr(estimator, "tree_"):
        return float(estimator.tree_.node_count)
    if hasattr(estimator, "estimators_"):
        total = 0.0
        for sub in np.asarray(estimator.estimators_).ravel():
            total += _n_nodes(sub)
        return total
    return float("nan")


def probe_specs(n_jobs_model: int) -> list[tuple[str, str, object]]:
    """Configurazioni sonda: gli angoli costosi delle griglie candidate e un
    punto intermedio per ciascun modello, da cui interpolare il resto.

    L'implementazione esterna di gradient boosting riceve un solo processo,
    perche' dentro la ricerca su griglia il parallelismo e' gia' speso sulle
    configurazioni: misurarla con tutti i processori sovrastimerebbe la sua
    velocita' relativa nelle condizioni in cui verra' effettivamente usata.
    """
    return [
        (
            "tree_unpruned",
            "Albero non potato",
            DecisionTreeRegressor(random_state=MODEL_SEED),
        ),
        (
            "tree_pruned",
            "Albero potato, ccp_alpha=10",
            DecisionTreeRegressor(random_state=MODEL_SEED, ccp_alpha=10.0),
        ),
        (
            "adaboost_max",
            "AdaBoost, 400 stadi, profondita' 4",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4, random_state=MODEL_SEED),
                n_estimators=400,
                learning_rate=0.2,
                random_state=MODEL_SEED,
            ),
        ),
        (
            "adaboost_min",
            "AdaBoost, 100 stadi, profondita' 2",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=2, random_state=MODEL_SEED),
                n_estimators=100,
                learning_rate=0.2,
                random_state=MODEL_SEED,
            ),
        ),
        (
            "gradient_boosting_max",
            "Gradient boosting, 600 stadi, profondita' 5",
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=5,
                random_state=MODEL_SEED,
            ),
        ),
        (
            "gradient_boosting_min",
            "Gradient boosting, 100 stadi, profondita' 2",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=MODEL_SEED,
            ),
        ),
        (
            "xgboost_max",
            "XGBoost, 600 stadi, profondita' 5",
            XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=5,
                objective="reg:squarederror",
                random_state=MODEL_SEED,
                n_jobs=n_jobs_model,
            ),
        ),
        (
            "xgboost_min",
            "XGBoost, 100 stadi, profondita' 2",
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                objective="reg:squarederror",
                random_state=MODEL_SEED,
                n_jobs=n_jobs_model,
            ),
        ),
    ]


def saturation_specs(n_jobs: int) -> list[tuple[str, str, object]]:
    """Insiemi per aggregazione su cui viene misurata la saturazione.

    Il bagging e la foresta a frazione unitaria sono lo stesso modello sotto due
    classi diverse: la coincidenza dei loro errori e' un controllo di
    correttezza, e la loro distanza dalla foresta decorrelata e' l'effetto che
    la famiglia serve a mostrare.
    """
    return [
        (
            "bagging",
            "Bagging di alberi non potati",
            BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=MODEL_SEED),
                bootstrap=True,
                random_state=MODEL_SEED,
                n_jobs=n_jobs,
                warm_start=True,
            ),
        ),
        (
            "forest_all_features",
            "Foresta, tutte le variabili candidate",
            RandomForestRegressor(
                max_features=1.0,
                random_state=MODEL_SEED,
                n_jobs=n_jobs,
                warm_start=True,
            ),
        ),
        (
            "forest_third_features",
            "Foresta, un terzo delle variabili candidate",
            RandomForestRegressor(
                max_features=0.33,
                random_state=MODEL_SEED,
                n_jobs=n_jobs,
                warm_start=True,
            ),
        ),
    ]


def run_probe(design, split, n_jobs_model: int) -> pd.DataFrame:
    """Adatta ogni configurazione sonda su una partizione e ne registra il costo."""
    X_train = design.X_train.iloc[split.train]
    y_train = design.y_train[split.train]
    X_valid = design.X_train.iloc[split.valid]
    y_valid = design.y_train[split.valid]

    records = []
    for key, label, estimator in probe_specs(n_jobs_model):
        pipeline = build_pipeline(estimator)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = pipeline.predict(X_valid)
        predict_seconds = time.perf_counter() - start

        model = pipeline.named_steps["model"]
        record = {
            "subset": design.subset,
            "key": key,
            "label": label,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "n_nodes": _n_nodes(model),
            "n_leaves": float(model.get_n_leaves()) if hasattr(model, "get_n_leaves") else np.nan,
        }
        record.update(regression_metrics(y_valid, y_pred))
        records.append(record)
        print(
            f"    {label:<45} {fit_seconds:7.1f} s   "
            f"nodi {record['n_nodes']:>12,.0f}   rmse {record['rmse']:.2f}"
        )
    return pd.DataFrame.from_records(records)


def run_saturation(design, split, checkpoints, n_jobs: int) -> pd.DataFrame:
    """Errore e costo al crescere del numero di alberi, per gli insiemi per aggregazione.

    Gli alberi sono aggiunti allo stesso insieme invece di ricostruirlo a ogni
    punto: l'intera curva costa quanto il solo adattamento con il numero massimo
    di alberi, e i punti sono annidati l'uno nell'altro, quindi la curva descrive
    la crescita di un unico insieme e non il confronto fra insiemi diversi.
    """
    X_train = design.X_train.iloc[split.train]
    y_train = design.y_train[split.train]
    X_valid = design.X_train.iloc[split.valid]
    y_valid = design.y_train[split.valid]

    records = []
    for key, label, estimator in saturation_specs(n_jobs):
        pipeline = build_pipeline(estimator)
        cumulative = 0.0
        for n_trees in checkpoints:
            pipeline.set_params(model__n_estimators=n_trees)

            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            cumulative += time.perf_counter() - start

            y_pred = pipeline.predict(X_valid)
            record = {
                "subset": design.subset,
                "key": key,
                "label": label,
                "n_estimators": n_trees,
                "fit_seconds_cumulative": cumulative,
                "n_nodes": _n_nodes(pipeline.named_steps["model"]),
            }
            record.update(regression_metrics(y_valid, y_pred))
            records.append(record)
            print(
                f"    {label:<45} {n_trees:>4} alberi   "
                f"rmse {record['rmse']:.3f}   nodi {record['n_nodes']:>12,.0f}   "
                f"{cumulative:6.1f} s"
            )
    return pd.DataFrame.from_records(records)


def run_subset(subset: str, cap: int | None, checkpoints, n_jobs: int) -> dict:
    design = build_design(subset, cap=cap)
    split = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=SEARCH_SEEDS)[0]

    print(f"\n=== {subset} ===")
    print(
        f"partizione di misura: {len(split.train):,} righe e "
        f"{len(np.unique(design.groups_train[split.train]))} motori in addestramento, "
        f"{len(split.valid):,} righe in verifica, {len(design.features)} variabili"
    )

    print("\n  costo per adattamento")
    probe = run_probe(design, split, n_jobs_model=1)

    print("\n  saturazione degli insiemi per aggregazione")
    saturation = run_saturation(design, split, checkpoints, n_jobs=n_jobs)

    return {"cost_probe": probe, "ensemble_saturation": saturation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument(
        "--max-trees",
        type=int,
        default=max(SATURATION_CHECKPOINTS),
        help="ultimo punto della curva di saturazione",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    checkpoints = [n for n in SATURATION_CHECKPOINTS if n <= args.max_trees]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        outputs = run_subset(subset, args.cap, checkpoints, args.n_jobs)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
