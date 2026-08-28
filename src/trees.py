"""Lettura dei modelli della famiglia ad albero.

Ruolo nel progetto
    Fornisce al blocco dei laboratori 9 e 10 le funzioni con cui se ne
    commentano i modelli, come `src.nonlinear` fa per il blocco che supera la
    linearita'. Non contiene stimatori ne' logica di valutazione: gli stimatori
    sono nel registro, la valutazione nel protocollo.

Cosa riceve
    Pipeline gia' adattate, oppure una pipeline non adattata e un insieme di
    partizioni per la misura che richiede un riaddestramento.

Cosa produce
    Tabelle di importanza delle variabili e il riepilogo strutturale
    dell'albero potato.

Le due misure di importanza
    Sono le due letture del laboratorio 10 e misurano cose diverse.

    La riduzione di impurita' e' calcolata durante la costruzione dell'albero:
    somma quanto ciascuna variabile ha ridotto l'errore quadratico nei nodi in
    cui e' stata usata, pesando per il numero di osservazioni che li
    attraversano. E' gratuita perche' e' un sottoprodotto dell'addestramento, ed
    e' distorta verso le variabili con molti valori distinti, che offrono piu'
    punti di taglio fra cui scegliere il migliore. Su questo dataset la
    distorsione ha un bersaglio preciso, il numero di ciclo, che e' un conteggio
    e assume percio' piu' valori distinti di qualunque lettura di sensore.
    Inoltre e' calcolata sui dati di addestramento: una variabile puo' risultare
    importante per come l'albero e' cresciuto e non per quanto serve a predire.

    L'importanza per permutazione misura invece di quanto peggiora l'errore
    quando i valori di una variabile vengono mescolati fra le righe, e va
    calcolata su dati che il modello non ha visto. Il laboratorio la calcola
    sull'insieme di verifica; qui l'insieme di verifica ufficiale e' chiuso fino
    alla chiusura della graduatoria, quindi la misura avviene sulla parte di
    verifica di ciascuna partizione del seme di ricerca, con il modello
    riaddestrato sulla parte di addestramento della stessa partizione. La media e
    la dispersione fra partizioni dicono anche quanto la lettura sia stabile.

    Cautela comune alle due misure: fra variabili correlate l'importanza si
    ripartisce, e per la permutazione si annulla, perche' mescolare una variabile
    lascia al modello l'informazione contenuta nelle sue sostitute. Una
    importanza bassa non e' quindi una prova che la variabile sia inutile.

Unita' dell'importanza per permutazione
    La misura e' calcolata sulla metrica di riferimento del progetto e riportata
    come aumento della radice dell'errore quadratico medio, quindi in cicli. Un
    valore di 2 significa che mescolare quella variabile costa due cicli di
    errore, ed e' leggibile accanto ai valori della tabella di confronto.

Convenzioni delle librerie sull'importanza per riduzione di impurita'
    Scikit-learn restituisce importanze gia' normalizzate a somma uno.
    L'implementazione esterna di gradient boosting non normalizza, e per
    impostazione predefinita restituisce il guadagno medio per divisione anziche'
    il guadagno totale: sotto lo stesso nome le due librerie riporterebbero
    quantita' diverse. Il registro chiede il guadagno totale, che e' l'analogo
    della grandezza di scikit-learn, e questo lettore normalizza comunque tutte
    le colonne a somma uno.

    Il bagging e' l'unico insieme di scikit-learn che non espone affatto
    l'importanza delle variabili, mentre foresta, AdaBoost e gradient boosting la
    espongono. Viene percio' ricostruita qui come media non pesata delle
    importanze dei suoi alberi, che e' la definizione stessa usata dalla foresta:
    l'importanza restituita da una foresta coincide cifra per cifra con la media
    delle importanze dei suoi alberi, quindi la ricostruzione non introduce una
    grandezza diversa da quella riportata sulle altre righe. La mappatura passa
    dagli indici delle colonne viste da ciascun albero, che con le impostazioni
    del progetto sono tutte, e resta scritta in forma generale per non dipendere
    in modo implicito da quelle impostazioni.

    AdaBoost fa media pesata per il peso di ciascuno stadio anziche' semplice.
    La differenza e' propria del modello e non viene uniformata: la quantita'
    riportata resta, per ogni riga, l'importanza che quella tecnica definisce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Metrica su cui e' misurata l'importanza per permutazione. E' la stessa su cui
# il progetto seleziona e ordina: una importanza misurata su una metrica diversa
# da quella del confronto risponderebbe a una domanda diversa da quella posta.
PERMUTATION_SCORING = "neg_root_mean_squared_error"


def _importances_from_members(model, n_features: int) -> np.ndarray:
    """Importanza di un insieme che non la espone, dalla media dei suoi alberi.

    Ogni albero riporta importanze relative alle sole colonne che ha visto, che
    `estimators_features_` identifica: i contributi vanno percio' riportati sulle
    posizioni originali prima di essere mediati.
    """
    members = list(np.asarray(model.estimators_, dtype=object).ravel())
    seen = getattr(model, "estimators_features_", None)

    total = np.zeros(n_features, dtype=float)
    for index, member in enumerate(members):
        member_importances = np.asarray(member.feature_importances_, dtype=float)
        if seen is None:
            total += member_importances
        else:
            total[np.asarray(seen[index], dtype=int)] += member_importances
    return total / len(members)


def impurity_importances(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Importanza per riduzione di impurita', normalizzata a somma uno.

    La colonna `zero` individua le variabili che il modello non ha mai usato per
    una divisione, ed e' la stessa convenzione con cui negli altri blocchi si
    contano i coefficienti annullati: rende la colonna del numero di variabili
    attive confrontabile lungo l'intera tabella del confronto.
    """
    model = pipeline.named_steps["model"]

    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        raw = _importances_from_members(model, len(feature_names))
    raw = np.asarray(raw, dtype=float).ravel()

    if len(raw) != len(feature_names):
        raise AssertionError(
            f"{len(raw)} importanze per {len(feature_names)} variabili: "
            f"la pipeline non conserva l'ordine delle colonne"
        )

    total = float(raw.sum())
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": raw / total if total > 0 else raw,
            "importance_raw": raw,
        }
    )
    frame["zero"] = np.isclose(frame["importance"], 0.0)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importances(
    estimator,
    design,
    splits,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Importanza per permutazione sulle parti di verifica delle partizioni indicate.

    Per ciascuna partizione il modello viene riaddestrato sulla sua parte di
    addestramento e l'importanza e' misurata sulla parte di verifica, che non ha
    partecipato all'addestramento. Le partizioni passate sono quelle del seme di
    ricerca: la misura non e' una stima di prestazione e non entra in
    graduatoria, quindi non richiede le quindici partizioni di confronto.

    Il parallelismo e' disattivato per impostazione predefinita. La funzione di
    libreria distribuisce le variabili sui processi e ne duplica il modello: su
    un insieme di trecento alberi non potati ogni copia occupa centinaia di
    megabyte, e il guadagno di tempo non compensa il rischio.
    """
    X, y = design.X_train, np.asarray(design.y_train)
    features = list(design.features)

    frames = []
    for split in splits:
        model = clone(estimator).fit(X.iloc[split.train], y[split.train])
        result = permutation_importance(
            model,
            X.iloc[split.valid],
            y[split.valid],
            scoring=PERMUTATION_SCORING,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=n_jobs,
        )
        frames.append(
            pd.DataFrame(
                {
                    "seed": split.seed,
                    "fold": split.fold,
                    "feature": features,
                    # Il punteggio e' orientato in modo che valori maggiori
                    # siano migliori, quindi la differenza fra punteggio di
                    # riferimento e punteggio dopo la permutazione e' gia'
                    # l'aumento dell'errore, in cicli.
                    "importance": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
        )

    per_fold = pd.concat(frames, ignore_index=True)
    summary = (
        per_fold.groupby("feature", sort=False)["importance"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "importance_mean", "std": "importance_fold_std"})
        .reset_index()
    )
    return summary.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def tree_summary(pipeline: Pipeline) -> dict:
    """Dimensione dell'albero potato: nodi, foglie, profondita'.

    Sono le quantita' con cui si legge quanto la potatura ha ridotto l'albero
    cresciuto per intero, che sulla matrice del progetto ha circa una foglia per
    riga di addestramento.
    """
    model = pipeline.named_steps["model"]
    return {
        "n_nodes": int(model.tree_.node_count),
        "n_leaves": int(model.get_n_leaves()),
        "depth": int(model.get_depth()),
    }