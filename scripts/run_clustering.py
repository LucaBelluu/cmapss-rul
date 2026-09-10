"""Raggruppamento delle traiettorie con i metodi non supervisionati del corso.

Ruolo nel progetto
    Produce il materiale con cui si commenta l'asimmetria fra i due sottoinsiemi
    in perimetro. FD001 ha un solo modo di guasto e FD003 ne ha due, e il
    confronto fra modelli mostra su FD003 traiettorie più lunghe, dispersione
    quasi doppia delle durate e punteggi migliori. La domanda che questo passo
    chiude è se la differenza sia visibile nella forma delle traiettorie
    guardandole senza il target, cioè se un metodo non supervisionato trovi su
    FD003 una struttura di gruppi che su FD001 non c'è.

    Il risultato è strumento di commento e non entra in graduatoria. Il task del
    progetto è di regressione e i metodi non supervisionati del laboratorio 12
    rientrano come strumenti di esplorazione.

Cosa riceve
    I dati grezzi attraverso la catena `src.data`, `src.target`, `src.design`.
    Nessun argomento obbligatorio. Non legge la graduatoria e non legge
    l'insieme di verifica ufficiale.

Cosa produce
    In `experiments/clustering/`, per ciascun sottoinsieme e per la versione che
    unisce i due:

    - `{NOME}_engine_features.csv`, le variabili per motore;
    - `{NOME}_cluster_scores.csv`, silhouette e accordo fra metodi al variare del
      numero di gruppi;
    - `{NOME}_cluster_labels.csv`, le etichette assegnate a ciascun motore, con
      la durata della traiettoria e le coordinate della proiezione;
    - `{NOME}_kmeans_seeds.csv`, la sensibilità di K-Means al seme;
    - `{NOME}_linkage.csv`, la matrice di aggregazione del dendrogramma.

    Per ciascun sottoinsieme produce inoltre `{SUBSET}_senza_durata_cluster_scores.csv`
    e `{SUBSET}_senza_durata_cluster_labels.csv`, descritti sotto.

Le due analisi
    Dentro ciascun sottoinsieme, che è l'analisi principale e risponde alla
    domanda sui modi di guasto. Qui non esiste etichetta di riferimento, perché
    il modo di guasto dei singoli motori non è distribuito con il dataset:
    l'indice di Rand corretto misura l'accordo fra due raggruppamenti diversi e
    non la loro correttezza, e il numero di gruppi si legge sulla silhouette.

    Sui duecento motori dei due sottoinsiemi uniti, come coda. Qui l'appartenenza
    al sottoinsieme è un'etichetta esterna vera, ed è l'unico punto del
    progetto in cui l'indice di Rand corretto ha il significato che ha nel
    laboratorio. La domanda è se un metodo non supervisionato separi le due
    popolazioni senza sapere da quale file provengono.

    L'unione usa le sole variabili presenti in entrambi i sottoinsiemi:
    `sensor_10` è costante su FD001 e non lo è su FD003, quindi entra nella
    matrice del secondo e non in quella dell'unione.

Controllo sulla durata
    La durata della traiettoria è una delle variabili del raggruppamento e ne è
    anche la lettura più immediata, quindi una separazione che si legga sulle
    durate potrebbe essere prodotta dalla durata stessa invece che dallo stato
    dei sensori. Il controllo ripete l'analisi sulle sole letture dei sensori e
    ne registra silhouette ed etichette: se la partizione non cambia, la
    separazione è prodotta dai sensori e la differenza di durata ne è una
    conseguenza.

    Del controllo vengono scritti punteggi ed etichette e non l'intera sequenza:
    le variabili per motore sono quelle già scritte meno una colonna, e la
    matrice di aggregazione serve al dendrogramma dell'analisi principale.

Come si lancia
    python -m scripts.run_clustering
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.clustering import (
    cluster_labels,
    cluster_scores,
    engine_features,
    linkage_matrix,
    projection,
    seed_stability,
    standardize,
)
from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "clustering"

# Suffisso con cui il controllo sulla durata compare nei nomi dei file e nella
# colonna `insieme` dei punteggi.
VARIANT = "senza_durata"

# Artefatti del controllo sulla durata. Gli altri tre prodotti da `analyse` non
# vengono scritti: le variabili per motore sono quelle dell'analisi principale
# meno una colonna, e la matrice di aggregazione serve al solo dendrogramma.
VARIANT_OUTPUTS = ("cluster_scores", "cluster_labels")


def analyse(
    name: str,
    features: pd.DataFrame,
    reference: np.ndarray | None = None,
    durations: pd.Series | None = None,
) -> dict:
    """Esegue la sequenza completa su una tabella di variabili per motore.

    `durations` esiste perché il controllo sulla durata passa una matrice che
    quella colonna non contiene, mentre le etichette la riportano comunque: è la
    quantità con cui i gruppi vengono descritti, anche quando non partecipa al
    calcolo delle distanze.
    """
    matrix = standardize(features)
    coordinates, explained = projection(matrix)

    scores = cluster_scores(matrix, labels_reference=reference)
    scores.insert(0, "insieme", name)

    labels = cluster_labels(matrix, features.index)
    labels.insert(0, "durata", features["durata"] if durations is None else durations)
    labels.insert(1, "pc1", coordinates[:, 0])
    labels.insert(2, "pc2", coordinates[:, 1])
    if reference is not None:
        labels.insert(0, "sottoinsieme", reference)

    seeds = seed_stability(matrix)
    seeds.insert(0, "insieme", name)

    print(f"\n=== {name} ===")
    print(
        f"{features.shape[0]} motori, {features.shape[1]} variabili, "
        f"varianza trattenuta dalle prime due componenti "
        f"{explained.sum():.3f}"
    )
    columns = ["n_clusters", "metodo", "linkage", "silhouette", "ari_vs_kmeans"]
    if reference is not None:
        columns.append("ari_vs_reference")
    columns += ["dimensione_minima", "dimensione_massima"]
    print(scores[columns].round(3).to_string(index=False))

    print("\nstabilità di K-Means al variare del seme")
    print(
        seeds.pivot_table(
            index="n_clusters", columns="seme", values="ari_vs_seme_riferimento"
        )
        .round(3)
        .to_string()
    )

    return {
        "engine_features": features.reset_index(),
        "cluster_scores": scores,
        "cluster_labels": labels.reset_index(),
        "kmeans_seeds": seeds,
        "linkage": linkage_matrix(matrix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_subset = {}
    for subset in args.subsets:
        design = build_design(subset)
        features = engine_features(design)
        per_subset[subset] = features
        for suffix, frame in analyse(subset, features).items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{suffix}.csv", index=False)

        # Controllo sulla durata, sullo stesso sottoinsieme e con le stesse
        # impostazioni: cambia soltanto la matrice su cui le distanze sono
        # calcolate.
        variant = analyse(
            f"{subset}_{VARIANT}",
            features.drop(columns=["durata"]),
            durations=features["durata"],
        )
        for suffix in VARIANT_OUTPUTS:
            variant[suffix].to_csv(OUTPUT_DIR / f"{subset}_{VARIANT}_{suffix}.csv", index=False)

    if len(per_subset) < 2:
        print("\nunione non prodotta: richiede entrambi i sottoinsiemi")
        return

    # L'unione usa le sole variabili comuni. Gli identificativi di unità si
    # ripetono fra sottoinsiemi (1..100 in entrambi), quindi vengono resi
    # distinti prima di concatenare: senza questo passaggio due motori diversi
    # occuperebbero la stessa riga dell'indice.
    common = sorted(
        set.intersection(*(set(f.columns) for f in per_subset.values())),
        key=list(next(iter(per_subset.values())).columns).index,
    )
    pieces = []
    origin = []
    for subset, features in per_subset.items():
        piece = features[common].copy()
        piece.index = [f"{subset}_{u}" for u in piece.index]
        pieces.append(piece)
        origin.extend([subset] * len(piece))

    pooled = pd.concat(pieces)
    pooled.index.name = "unit"
    for suffix, frame in analyse("unione", pooled, reference=np.array(origin)).items():
        frame.to_csv(OUTPUT_DIR / f"unione_{suffix}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()