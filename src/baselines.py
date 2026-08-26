"""Baseline di riferimento per la lettura dei risultati.

Ruolo nel progetto
    Fornisce i due termini di paragone rispetto ai quali si legge il guadagno
    di ogni modello del confronto. Non sono modelli in gara: sono il pavimento
    sotto cui un risultato non ha significato.

Cosa riceve
    Nulla dal chiamante se non, per la seconda baseline, il nome della colonna
    del numero di ciclo.

Cosa produce
    Pipeline pronte per il protocollo di valutazione.

Le due baseline
    La predizione costante restituisce la media del target calcolata sulla
    parte di addestramento di ciascuna partizione. Il suo errore quadratico
    medio coincide, a meno della correzione per i gradi di liberta', con la
    deviazione standard del target: e' il pavimento assoluto, e un modello che
    non lo batte non ha appreso nulla.

    La regressione sul solo numero di ciclo e' il pavimento informativo. Su
    traiettorie che arrivano tutte al guasto la vita utile residua e' la
    differenza fra durata e ciclo corrente, e le durate hanno dispersione
    limitata: un conteggio dei cicli predice quindi il target con un errore non
    trascurabile senza usare alcuna lettura di sensore. Il guadagno di un
    modello va letto rispetto a questa baseline e non rispetto alla precedente,
    altrimenti si attribuisce ai sensori una capacita' predittiva che proviene
    dal solo numero di cicli percorsi.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data import CYCLE_COL
from src.pipeline import build_pipeline


def constant_baseline() -> Pipeline:
    """Predizione costante pari alla media del target di addestramento."""
    # La standardizzazione e' disattivata: non ha effetto su un predittore
    # costante e la sua presenza renderebbe meno leggibile l'oggetto.
    return build_pipeline(DummyRegressor(strategy="mean"), scale=False)


def cycle_only_baseline() -> Pipeline:
    """Regressione lineare sul solo numero di ciclo."""
    return build_pipeline(LinearRegression(), columns=[CYCLE_COL])


def all_baselines() -> dict[str, Pipeline]:
    return {
        "baseline_costante": constant_baseline(),
        "baseline_solo_ciclo": cycle_only_baseline(),
    }