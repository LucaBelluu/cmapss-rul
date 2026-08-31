"""Composizione dei quattro blocchi del confronto in un'unica graduatoria.

Ruolo nel progetto
    I quattro blocchi hanno prodotto ciascuno la propria tabella, valutata sulle
    stesse 15 partizioni sotto lo stesso protocollo. Questo modulo le compone in
    una graduatoria sola e verifica prima che la composizione sia legittima,
    cioe' che le partizioni siano davvero identiche nei quattro blocchi. Non
    addestra modelli, non esegue lavoro computazionale e non legge in alcun modo
    l'insieme di verifica ufficiale.

Cosa riceve
    Gli artefatti gia' su disco: `experiments/<blocco>/{SUBSET}_comparison.csv`
    e `experiments/<blocco>/{SUBSET}_cv_folds.csv`.

Cosa produce
    DataFrame con la verifica di identita' delle partizioni, la graduatoria
    complessiva e il confronto appaiato fold per fold. Non scrive su disco: la
    persistenza e' compito degli script di orchestrazione.

Perche' la verifica delle partizioni precede la graduatoria
    Comporre quattro tabelle prodotte da esecuzioni diverse presuppone che i
    punteggi siano stati calcolati sugli stessi motori e sulle stesse righe. Il
    presupposto non e' garantito dal fatto che il codice del protocollo sia
    unico: basterebbe una versione diversa della catena dati fra un'esecuzione e
    l'altra per invalidarlo. La verifica esiste perche' le due baseline sono
    ricalcolate in ogni blocco, quindi ogni blocco contiene quattro misure
    indipendenti della stessa quantita' sulle stesse partizioni. Il confronto
    dei conteggi di righe di ciascun fold e' la firma della partizione;
    l'uguaglianza dei punteggi la corrobora. Se la verifica fallisce, la
    graduatoria complessiva non e' costruibile e i blocchi restano leggibili
    solo separatamente.

Confronto appaiato
    Le 15 partizioni sono le stesse per tutti i modelli, quindi la differenza
    fra due modelli si puo' calcolare fold per fold invece che confrontando due
    medie con le rispettive dispersioni. La difficolta' del fold, che e' la
    componente dominante della dispersione riportata in tabella, e' comune ai
    due modelli e si elide nella differenza.

    La media delle differenze appaiate coincide per costruzione con la
    differenza delle medie: non e' li' che sta l'informazione aggiuntiva. Quella
    sta nella dispersione della differenza, che e' molto piu' piccola della
    dispersione dei singoli punteggi quando i due modelli sbagliano sugli stessi
    fold, e nella concordanza del segno, cioe' in quanti fold su 15 lo stesso
    modello risulta migliore.

    La lettura e' fuori dal materiale del corso e va dichiarata come tale.
    Resta una lettura descrittiva: non viene prodotta alcuna statistica test e
    il rapporto fra media e dispersione della differenza non e' convertibile in
    un livello di significativita', perche' i 15 fold condividono le righe di
    addestramento e non sono osservazioni indipendenti. La regola di lettura
    adottata nei quattro blocchi resta quella fissata dal protocollo e questa
    lettura non la sostituisce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.experiment import gap_in_dispersions

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Cartelle dei quattro blocchi, nell'ordine in cui sono stati eseguiti, con
# l'etichetta con cui compaiono nella graduatoria.
BLOCKS: dict[str, str] = {
    "linear_models": "Modelli lineari",
    "nonlinear_models": "Superamento della linearita'",
    "tree_models": "Famiglia ad albero",
    "kernel_models": "Metodi a margine e reti",
}

BASELINE_KEYS = ("baseline_costante", "baseline_solo_ciclo")

# Blocco da cui le baseline entrano nella graduatoria. La scelta e' arbitraria
# per costruzione: la verifica delle partizioni ha appena stabilito che le
# quattro copie coincidono, e se non coincidessero la graduatoria non verrebbe
# prodotta affatto.
BASELINE_SOURCE = "kernel_models"

FOLD_KEYS = ["seed", "fold"]

# Tolleranza sullo scarto fra le copie della stessa baseline in blocchi diversi.
# Le esecuzioni sono deterministiche e lo scarto atteso e' nullo; il margine
# assorbe differenze di ordine delle operazioni in virgola mobile fra versioni
# della catena, non differenze di partizione, che si manifesterebbero
# sull'ordine dell'unita'.
PARTITION_TOLERANCE = 1e-6


def load_block(subset: str, block: str, name: str) -> pd.DataFrame:
    """Legge un artefatto di un blocco, annotandone la provenienza."""
    path = EXPERIMENTS_DIR / block / f"{subset}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"manca {path}: la graduatoria complessiva richiede che tutti i "
            f"blocchi siano stati eseguiti"
        )
    frame = pd.read_csv(path)
    frame.insert(0, "block", block)
    return frame


def all_comparisons(subset: str) -> pd.DataFrame:
    return pd.concat([load_block(subset, b, "comparison") for b in BLOCKS], ignore_index=True)


def all_folds(subset: str) -> pd.DataFrame:
    return pd.concat([load_block(subset, b, "cv_folds") for b in BLOCKS], ignore_index=True)


def check_partitions(subset: str, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Verifica che i quattro blocchi abbiano usato le stesse partizioni.

    Confronta, per ciascuna baseline e ciascun blocco, l'insieme dei fold, i
    conteggi di righe e di motori di ogni fold e il punteggio ottenuto, contro
    la copia del blocco di riferimento. Solleva un errore se qualcosa non
    coincide: e' il presupposto della graduatoria complessiva, e un presupposto
    che fallisce va fermato qui e non attenuato in una nota.
    """
    folds = all_folds(subset) if folds is None else folds
    baselines = folds[folds["model"].isin(BASELINE_KEYS)]
    if baselines.empty:
        raise AssertionError(f"{subset}: nessuna baseline negli artefatti dei blocchi")

    counts = ["n_train_rows", "n_valid_rows", "n_train_units", "n_valid_units"]
    records = []
    for key, group in baselines.groupby("model", sort=True):
        reference = group[group["block"] == BASELINE_SOURCE].set_index(FOLD_KEYS).sort_index()
        if reference.empty:
            raise AssertionError(f"{subset}: {key} assente in {BASELINE_SOURCE}")
        for block, other in group.groupby("block", sort=True):
            other = other.set_index(FOLD_KEYS).sort_index()
            if not reference.index.equals(other.index):
                raise AssertionError(
                    f"{subset}: {key} in {block} ha partizioni diverse da {BASELINE_SOURCE}"
                )
            identical_counts = bool((reference[counts] == other[counts]).all().all())
            gap = float((reference["rmse"] - other["rmse"]).abs().max())
            records.append(
                {
                    "subset": subset,
                    "model": key,
                    "block": block,
                    "n_partizioni": len(other),
                    "conteggi_identici": identical_counts,
                    "scarto_max_rmse": gap,
                }
            )

    report = pd.DataFrame(records)
    broken = report[~report["conteggi_identici"]]
    if not broken.empty:
        raise AssertionError(
            f"{subset}: conteggi di riga diversi fra blocchi, partizioni non identiche\n"
            f"{broken.to_string(index=False)}"
        )
    drifted = report[report["scarto_max_rmse"] > PARTITION_TOLERANCE]
    if not drifted.empty:
        raise AssertionError(
            f"{subset}: la stessa baseline ottiene punteggi diversi in blocchi diversi\n"
            f"{drifted.to_string(index=False)}"
        )
    return report


def _drop_duplicate_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Tiene una sola copia delle baseline, ricalcolate in ogni blocco."""
    is_baseline = frame["model"].isin(BASELINE_KEYS)
    keep = ~is_baseline | (frame["block"] == BASELINE_SOURCE)
    return frame[keep].reset_index(drop=True)


def overall_folds(subset: str, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Metriche per fold di tutti i modelli dei quattro blocchi, senza duplicati."""
    folds = all_folds(subset) if folds is None else folds
    folds = _drop_duplicate_baselines(folds)
    duplicated = folds.duplicated(subset=["model"] + FOLD_KEYS)
    if duplicated.any():
        names = sorted(folds.loc[duplicated, "model"].unique())
        raise AssertionError(
            f"{subset}: gli identificativi {names} compaiono in piu' di un blocco; "
            f"la graduatoria richiede che ogni modello abbia una sola riga"
        )
    return folds


def overall_ranking(subset: str, comparisons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Graduatoria complessiva sui quattro blocchi, ordinata sulla metrica di riferimento.

    La colonna del divario dalla riga migliore e' ricalcolata rispetto al primo
    posto complessivo e non a quello del blocco di provenienza, con la stessa
    definizione usata dalle tabelle dei singoli blocchi.

    La colonna `n_configurations` viene mantenuta perche' e' la quantita' con
    cui si legge la distorsione ottimistica di ciascuna riga: la
    cross-validation non e' annidata, e la distorsione cresce con il numero di
    configurazioni valutate sulle stesse partizioni su cui il punteggio e' poi
    riportato. Le righe della graduatoria non sono a parita' di questo fattore.
    """
    table = _drop_duplicate_baselines(all_comparisons(subset) if comparisons is None else comparisons)
    table["blocco"] = table["block"].map(BLOCKS)
    table = table.sort_values("rmse_mean").reset_index(drop=True)
    table["divario_in_dispersioni"] = gap_in_dispersions(table["rmse_mean"], table["rmse_std"])

    columns = [
        "subset",
        "blocco",
        "label",
        "model",
        "config",
        "rmse_mean",
        "rmse_std",
        "divario_in_dispersioni",
        "mae_mean",
        "r2_mean",
        "n_configurations",
        "n_fit",
    ]
    return table[[c for c in columns if c in table.columns]]


def paired_differences(
    folds: pd.DataFrame,
    reference: str,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Differenze fold per fold di ciascun modello rispetto a un modello di riferimento.

    Valore positivo: il modello sbaglia piu' del riferimento su quel fold.

    `differenza_media` coincide con la differenza fra le medie riportate in
    graduatoria ed e' inclusa come controllo di coerenza. Le colonne che portano
    informazione nuova sono `differenza_std`, che e' molto minore della
    dispersione dei punteggi quando i due modelli sbagliano sugli stessi fold, e
    `fold_peggiori`, che conta su quanti fold il segno si conferma.

    `differenza_in_dispersioni` e' il rapporto fra le due precedenti. Non e' una
    statistica test e non e' convertibile in un livello di significativita': i
    fold condividono le righe di addestramento e non sono indipendenti.
    """
    wide = folds.pivot_table(index=FOLD_KEYS, columns="model", values="rmse")
    if reference not in wide.columns:
        raise KeyError(f"modello di riferimento {reference} assente dalle metriche per fold")

    models = list(wide.columns) if models is None else list(models)
    labels = folds.drop_duplicates("model").set_index("model")

    records = []
    for model in models:
        difference = wide[model] - wide[reference]
        std = float(difference.std(ddof=1))
        records.append(
            {
                "model": model,
                "riferimento": reference,
                "rmse_mean": float(wide[model].mean()),
                "rmse_std": float(wide[model].std(ddof=1)),
                "differenza_media": float(difference.mean()),
                "differenza_std": std,
                "differenza_in_dispersioni": float(abs(difference.mean()) / std) if std else np.nan,
                "fold_peggiori": int((difference > 0).sum()),
                "n_fold": int(len(difference)),
                "differenza_min": float(difference.min()),
                "differenza_max": float(difference.max()),
            }
        )

    table = pd.DataFrame(records).sort_values("rmse_mean").reset_index(drop=True)
    if "block" in labels.columns:
        table.insert(1, "blocco", table["model"].map(labels["block"]).map(BLOCKS))
    return table


def best_model(ranking: pd.DataFrame) -> pd.Series:
    """Prima riga della graduatoria fra i modelli, escluse le baseline."""
    models = ranking[~ranking["model"].isin(BASELINE_KEYS)]
    if models.empty:
        raise AssertionError("la graduatoria non contiene modelli")
    return models.iloc[0]
