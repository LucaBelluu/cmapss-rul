"""Calcolo e salvataggio degli aggregati esplorativi dei quattro sottoinsiemi.

Ruolo nel progetto
    Orchestrazione della fase esplorativa. Carica i quattro sottoinsiemi,
    applica le funzioni di `src.explore` e salva i risultati come file CSV. È
    l'unico punto in cui l'esplorazione tocca i dati grezzi: il notebook di
    esplorazione legge questi file e non ricalcola nulla.

Cosa riceve
    I file grezzi in `data/raw/`.

Cosa produce
    In `experiments/exploration/`, otto file CSV:

        trajectory_lengths.csv      durata di ogni traiettoria, training e test
        operating_conditions.csv    regimi di funzionamento distinti
        variable_summary.csv        statistiche di sensori e impostazioni
        variable_summary_by_condition.csv   le stesse, dentro ogni regime
        target_correlations.csv     correlazione di ogni variabile con la RUL
        target_correlations_by_phase.csv    la stessa, divisa nelle due fasi di vita
        variable_correlation_matrix.csv     correlazione tra sensori
        sensor_traces.csv           traiettorie complete di cinque unità

    e un resoconto sintetico su standard output.

    La cartella `experiments/` non è versionata: i file sono rigenerabili
    lanciando di nuovo questo script. Gli artefatti destinati alla consegna
    (figure e tabella riassuntiva) sono prodotti dal notebook di esplorazione a
    partire da questi file.

Perimetro
    L'esplorazione copre tutti e quattro i sottoinsiemi, anche quelli esclusi
    dagli esperimenti: la loro caratterizzazione è ciò che documenta il criterio
    con cui il perimetro sperimentale è stato definito.

Esecuzione
    Dalla radice della repository:

        python -m scripts.run_exploration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT, SUBSETS, load_subset
from src.explore import (
    operating_conditions,
    sensor_traces,
    target_correlations,
    target_correlations_by_phase,
    trajectory_lengths,
    variable_correlation_matrix,
    variable_summary,
    variable_summary_by_condition,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exploration"


def build_tables() -> dict[str, pd.DataFrame]:
    """Calcola tutti gli aggregati, concatenando i quattro sottoinsiemi."""
    collected: dict[str, list[pd.DataFrame]] = {
        "trajectory_lengths": [],
        "operating_conditions": [],
        "variable_summary": [],
        "variable_summary_by_condition": [],
        "target_correlations": [],
        "target_correlations_by_phase": [],
        "variable_correlation_matrix": [],
        "sensor_traces": [],
    }

    for name in SUBSETS:
        print(f"Elaborazione {name}")
        subset = load_subset(name)
        train = subset.train

        collected["trajectory_lengths"].append(trajectory_lengths(train, name, "train"))
        collected["trajectory_lengths"].append(
            trajectory_lengths(subset.test, name, "test")
        )
        collected["operating_conditions"].append(operating_conditions(train, name))
        collected["variable_summary"].append(variable_summary(train, name))
        collected["variable_summary_by_condition"].append(
            variable_summary_by_condition(train, name)
        )
        collected["target_correlations"].append(target_correlations(train, name))
        collected["target_correlations_by_phase"].append(
            target_correlations_by_phase(train, name)
        )
        collected["variable_correlation_matrix"].append(
            variable_correlation_matrix(train, name)
        )
        collected["sensor_traces"].append(sensor_traces(train, name))

    return {key: pd.concat(parts, ignore_index=True) for key, parts in collected.items()}


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"  scritto {path.relative_to(PROJECT_ROOT)}  ({len(table)} righe)")


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Resoconto compatto, pensato per essere letto a colpo d'occhio."""
    lengths = tables["trajectory_lengths"]
    conditions = tables["operating_conditions"]
    summary = tables["variable_summary"]
    correlations = tables["target_correlations"]

    print("\nDurata delle traiettorie di training (cicli)")
    train_lengths = lengths[lengths["split"] == "train"]
    print(
        train_lengths.groupby("subset")["length"]
        .agg(["count", "min", "median", "mean", "max", "std"])
        .round(1)
        .to_string()
    )

    print("\nDurata delle traiettorie di test osservate (cicli)")
    test_lengths = lengths[lengths["split"] == "test"]
    print(
        test_lengths.groupby("subset")["length"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(1)
        .to_string()
    )

    print("\nRegimi di funzionamento distinti")
    print(conditions.groupby("subset").size().rename("n_condizioni").to_string())

    print("\nVariabili costanti sull'intero insieme di training")
    for name in sorted(summary["subset"].unique()):
        block = summary[(summary["subset"] == name) & summary["constant"]]
        listed = ", ".join(block["variable"]) if len(block) else "nessuna"
        print(f"  {name}: {len(block)} -> {listed}")

    print("\nVariabili con meno di dieci valori distinti")
    for name in sorted(summary["subset"].unique()):
        block = summary[(summary["subset"] == name) & (summary["n_unique"] < 10)]
        listed = ", ".join(
            f"{row.variable}({row.n_unique})" for row in block.itertuples()
        )
        print(f"  {name}: {len(block)} -> {listed if listed else 'nessuna'}")

    print("\nCorrelazione con la RUL nelle due fasi di vita")
    by_phase = tables["target_correlations_by_phase"]
    for name in sorted(by_phase["subset"].unique()):
        best = (
            correlations[correlations["subset"] == name]
            .dropna(subset=["abs_pearson"])
            .sort_values("abs_pearson", ascending=False)
            .head(5)["variable"]
            .tolist()
        )
        block = by_phase[(by_phase["subset"] == name) & by_phase["variable"].isin(best)]
        pivot = block.pivot(index="variable", columns="phase", values="pearson").loc[best]
        listed = ", ".join(
            f"{variable} {row.oltre_soglia:+.2f} -> {row.entro_soglia:+.2f}"
            for variable, row in pivot.iterrows()
        )
        print(f"  {name}: {listed}")

    print("\nCinque variabili più correlate con la RUL, in valore assoluto")
    for name in sorted(correlations["subset"].unique()):
        block = (
            correlations[correlations["subset"] == name]
            .sort_values("abs_pearson", ascending=False)
            .head(5)
        )
        listed = ", ".join(
            f"{row.variable} {row.pearson:+.2f}/{row.spearman:+.2f}"
            for row in block.itertuples()
        )
        print(f"  {name}: {listed}")


def main() -> int:
    tables = build_tables()
    print()
    save_tables(tables, OUTPUT_DIR)
    print_summary(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())