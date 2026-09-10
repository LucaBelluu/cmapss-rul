"""Controllo di sensibilità della graduatoria alla soglia di censura del target.

Ruolo nel progetto
    La censura del target a 125 cicli è un'ipotesi di modellazione fissata a
    priori, non una quantità misurata, e i valori assoluti di tutte le metriche
    dipendono da essa. Questo script misura se l'ordine fra le famiglie di
    modelli dipenda dalla stessa ipotesi, rivalutando un modello per famiglia
    sulle stesse partizioni con la censura disattivata.

Cosa riceve
    La graduatoria in `experiments/final/`, il registro dei modelli e i dati
    grezzi attraverso la catena `src.data`, `src.target`, `src.design`. Nessun
    argomento obbligatorio.

Cosa produce
    In `experiments/final/`, per ciascun sottoinsieme:

    - `{SUBSET}_censoring_folds.csv`, le metriche di ogni modello su ogni
      partizione nei due regimi;
    - `{SUBSET}_censoring_sensitivity.csv`, una riga per modello e regime con
      media e dispersione sulle 15 partizioni e la posizione nel regime.

    L'insieme di verifica ufficiale non viene letto.

Perimetro
    Un modello per famiglia: Ridge per i modelli lineari, il modello additivo
    generalizzato per i modelli non lineari additivi, la foresta casuale per la
    famiglia ad albero, il percettrone multistrato per i metodi a margine e le
    reti. Sono, in ciascuna famiglia, la riga meglio piazzata su FD001, e sono
    gli stessi sui due sottoinsiemi perché le due repliche restino confrontabili.

    Le due baseline entrano nel controllo insieme ai modelli. Sotto censura
    disattivata la predizione costante restituisce la deviazione standard del
    target non censurato, che è la scala su cui vanno letti gli errori di quel
    regime: senza quel riferimento l'incomparabilità dei valori assoluti fra i
    due regimi resterebbe un'affermazione invece che una misura.

La configurazione resta quella selezionata sotto censura
    La ricerca su griglia non viene rifatta nel regime senza censura. Rifarla
    equivarrebbe a condurre un secondo confronto completo su una diversa
    definizione del target, che il progetto ha scartato per costo quando la
    definizione è stata fissata.

    La conseguenza va dichiarata e la lettura che ne segue è a senso unico. Ogni
    configurazione è stata scelta per un target di scala diversa da quello su
    cui viene qui valutata, quindi una famiglia il cui ottimo si sposta molto
    risulta svantaggiata. Se l'ordine fra famiglie regge nonostante questo, il
    risultato è solido; se si inverte, non se ne può concludere che la famiglia
    sia peggiore sotto il target non censurato, perché l'inversione può essere
    prodotta dalla configurazione congelata.

Come si leggono i due regimi
    Non sull'errore. La censura tronca il target a 125 cicli e la sua rimozione
    ne aumenta dispersione ed escursione, quindi l'errore quadratico medio
    cresce per costruzione in ogni riga e i due regimi non sono confrontabili in
    valore assoluto. Ciò che si confronta è l'ordine dentro ciascun regime, e
    il coefficiente di determinazione, che è adimensionale e rapporta l'errore
    alla variabilità disponibile in quel regime.

Controllo incorporato
    Il regime con censura ripete una misura già in graduatoria e deve
    riprodurla. È lo stesso controllo di fedeltà della ricostruzione usato dal
    diagnostico sul seme dello stimatore, e vale qui per la ragione ulteriore che
    le due esecuzioni devono differire per la sola definizione del target: se il
    regime censurato non riproduce la graduatoria, la differenza osservata
    nell'altro regime non è attribuibile alla censura.

Come si lancia
    python -m scripts.run_censoring_sensitivity --subsets FD001 --models ridge
    python -m scripts.run_censoring_sensitivity
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.protocol import (
    COMPARISON_SEEDS,
    N_SPLITS,
    check_no_group_leakage,
    evaluate,
    make_splits,
)
from src.selected import load_ranking, rebuild
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final"

# Un modello per famiglia, nell'ordine dei blocchi del confronto.
FAMILY_MODELS = ("ridge", "gam", "random_forest", "mlp")
BASELINES = ("baseline_solo_ciclo", "baseline_costante")

REGIMES = {"censurato": RUL_CAP, "non censurato": None}

# Tolleranza sulla riproduzione del punteggio in graduatoria nel regime
# censurato. Assorbe le differenze di somma in virgola mobile fra esecuzioni,
# non una ricostruzione infedele.
REPRODUCTION_TOLERANCE = 1e-6


def build_regimes(subset: str) -> dict[str, object]:
    """Le due matrici di progetto, che devono differire per il solo target.

    L'uguaglianza delle variabili esplicative e degli identificativi di unità è
    verificata e non supposta: se le due matrici differissero anche solo
    nell'ordine delle righe, le partizioni non sarebbero le stesse e il confronto
    fra regimi misurerebbe due cose insieme.
    """
    designs = {name: build_design(subset, cap=cap) for name, cap in REGIMES.items()}
    reference = designs["censurato"]
    for name, design in designs.items():
        if not design.X_train.equals(reference.X_train):
            raise AssertionError(f"{subset}: la matrice del regime {name} non coincide")
        if not np.array_equal(design.groups_train, reference.groups_train):
            raise AssertionError(f"{subset}: le unità del regime {name} non coincidono")
    return designs


def run_subset(subset: str, models: list[str]) -> dict:
    designs = build_regimes(subset)
    reference = designs["censurato"]
    splits = make_splits(reference.groups_train, n_splits=N_SPLITS, seeds=COMPARISON_SEEDS)
    check_no_group_leakage(reference.groups_train, splits)

    ranking = load_ranking(subset)
    print(f"\n=== {subset} ===")
    for name, design in designs.items():
        y = design.y_train
        print(
            f"regime {name}: target da {int(y.min())} a {int(y.max())} cicli, "
            f"media {y.mean():.1f}, dispersione {y.std(ddof=1):.2f}"
        )

    fold_frames = []
    records = []
    for key in models:
        row = ranking[ranking["model"] == key]
        if row.empty:
            print(f"[{key}] assente dalla graduatoria, saltato")
            continue
        row = row.iloc[0]

        # La ricostruzione parte sempre dalla matrice censurata, che è quella
        # che ha prodotto la configurazione registrata. Lo stimatore non dipende
        # dal target, quindi è lo stesso oggetto nei due regimi.
        estimator = rebuild(key, row["config"], reference)
        print(f"\n[{key}] {row['config']}")

        for regime, design in designs.items():
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fold_metrics = evaluate(
                    estimator,
                    design.X_train,
                    design.y_train,
                    design.groups_train,
                    splits,
                )
            seconds = time.perf_counter() - start
            n_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

            fold_metrics.insert(0, "model", key)
            fold_metrics.insert(1, "regime", regime)
            fold_frames.append(fold_metrics)

            record = {
                "subset": subset,
                "regime": regime,
                "model": key,
                "label": row["label"],
                "config": row["config"],
            }
            for metric in ("rmse", "mae", "r2"):
                record[f"{metric}_mean"] = float(fold_metrics[metric].mean())
                record[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
            record["convergence_warnings"] = n_warnings
            record["seconds"] = seconds
            records.append(record)

            flag = f", {n_warnings} avvisi su {len(splits)} stime" if n_warnings else ""
            print(
                f"    {regime}: rmse {record['rmse_mean']:.3f} ± {record['rmse_std']:.3f}, "
                f"r2 {record['r2_mean']:.4f} ({seconds:.1f} s{flag})"
            )

            if regime == "censurato":
                gap = abs(record["rmse_mean"] - float(row["rmse_mean"]))
                status = "coincide" if gap <= REPRODUCTION_TOLERANCE else "NON COINCIDE"
                print(
                    f"    controllo di riproduzione: {status} con la graduatoria "
                    f"({row['rmse_mean']:.6f}, scarto {gap:.2e})"
                )

    table = pd.DataFrame.from_records(records)

    # La posizione è calcolata dentro ciascun regime e mai fra regimi: la scala
    # del target cambia, quindi gli errori dei due regimi non sono confrontabili
    # e l'unica quantità trasferibile è l'ordine.
    table["rango_nel_regime"] = table.groupby("regime")["rmse_mean"].rank(method="min").astype(int)
    table = table.sort_values(["regime", "rmse_mean"]).reset_index(drop=True)

    print("\nordine dentro ciascun regime")
    print(
        table[["regime", "model", "rmse_mean", "rmse_std", "r2_mean", "rango_nel_regime"]]
        .round(4)
        .to_string(index=False)
    )

    return {
        "censoring_folds": pd.concat(fold_frames, ignore_index=True),
        "censoring_sensitivity": table,
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
    args = parser.parse_args()

    partial = args.models is not None
    models = list(args.models) if partial else list(FAMILY_MODELS) + list(BASELINES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        outputs = run_subset(subset, models)
        if partial:
            print(f"\n{subset}: esecuzione parziale, artefatti non scritti")
            continue
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)
        print(f"\n{subset}: artefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()