"""Graduatoria complessiva del confronto, in cross-validation.

Ruolo nel progetto
    Compone in un'unica tabella i risultati dei quattro blocchi del confronto,
    dopo aver verificato che siano stati prodotti sulle stesse partizioni, e
    aggiunge la lettura appaiata fold per fold rispetto al modello migliore.
    E' il primo passo della fase di chiusura e chiude la graduatoria.

    Lo script non legge l'insieme di verifica ufficiale e non ne dipende in
    alcun modo. La separazione e' voluta: la graduatoria e la regola con cui
    verra' letto l'insieme di verifica sono fissate e depositate prima che
    esista il codice che quell'insieme lo legge.

Cosa riceve
    Gli artefatti dei quattro blocchi in `experiments/`. Nessun argomento
    obbligatorio. Non ricalcola nessun modello: legge tabelle gia' prodotte.

Cosa produce
    In `experiments/final/`, per ciascun sottoinsieme:

    - `{SUBSET}_partition_check.csv`, l'esito della verifica di identita' delle
      partizioni fra i quattro blocchi;
    - `{SUBSET}_ranking.csv`, la graduatoria complessiva;
    - `{SUBSET}_paired.csv`, il confronto appaiato di ogni modello rispetto al
      primo in graduatoria.

Regola di lettura della graduatoria
    L'ordinamento e' sulla radice dell'errore quadratico medio in
    cross-validation, media sulle 15 partizioni di confronto. Due modelli il cui
    divario e' inferiore a una dispersione fra fold non vengono ordinati: la
    graduatoria individua allora un gruppo di testa e non un vincitore. La
    lettura appaiata accompagna quella principale e non la sostituisce, ed e'
    fuori dal materiale del corso.

    I punteggi sono ottimisticamente distorti perche' la cross-validation non e'
    annidata, e la distorsione non e' uniforme fra le righe: cresce con il
    numero di configurazioni valutate, che nella graduatoria varia da una a
    centinaia di migliaia. La colonna che lo riporta e' parte della tabella e
    non una nota a margine.

Come si lancia
    python -m scripts.run_final_ranking
    python -m scripts.run_final_ranking --subsets FD001
"""

from __future__ import annotations

import argparse

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE
from src.final import (
    BASELINE_KEYS,
    best_model,
    check_partitions,
    overall_folds,
    overall_ranking,
    paired_differences,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final"


def run_subset(subset: str) -> dict:
    print(f"\n=== {subset} ===")

    check = check_partitions(subset)
    print("verifica delle partizioni fra i quattro blocchi")
    print(check.drop(columns=["subset"]).to_string(index=False))

    ranking = overall_ranking(subset)
    print("\ngraduatoria complessiva")
    print(
        ranking.drop(columns=["subset"])
        .assign(config=lambda f: f["config"].str.slice(0, 46))
        .to_string(index=False)
    )

    reference = best_model(ranking)
    print(f"\nprimo in graduatoria: {reference['label']} ({reference['model']})")

    folds = overall_folds(subset)
    paired = paired_differences(folds, reference=reference["model"])
    print("\nconfronto appaiato sulle 15 partizioni, rispetto al primo in graduatoria")
    print(
        paired[~paired["model"].isin(BASELINE_KEYS)][
            [
                "model",
                "blocco",
                "rmse_mean",
                "rmse_std",
                "differenza_media",
                "differenza_std",
                "differenza_in_dispersioni",
                "fold_peggiori",
            ]
        ].to_string(index=False)
    )

    return {"partition_check": check, "ranking": ranking, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        outputs = run_subset(subset)
        for name, frame in outputs.items():
            frame.to_csv(OUTPUT_DIR / f"{subset}_{name}.csv", index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
