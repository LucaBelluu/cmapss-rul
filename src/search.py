"""Ricerca degli iperparametri su griglia, sotto il protocollo del progetto.

Ruolo nel progetto
    Primo dei due stadi in cui si articola ogni esperimento. Cerca la
    configurazione di un modello valutando la griglia sulle partizioni del seme
    dedicato alla ricerca, e restituisce la griglia intera oltre al vincitore.
    Il secondo stadio, la rivalutazione della configurazione selezionata su
    tutte le partizioni di confronto, e' in `src.experiment`.

Cosa riceve
    Uno stimatore gia' composto con il pre-processing, la sua griglia, la
    matrice di progetto e le partizioni della ricerca.

Cosa produce
    La configurazione selezionata, la tabella completa della griglia con le tre
    metriche per ciascuna configurazione, e l'esito del controllo sui bordi.

Perche' la griglia intera e non il solo vincitore
    Il punteggio del vincitore non dice se il minimo sia netto o se la curva
    sia piatta, e la seconda situazione e' frequente sui modelli regolarizzati.
    La forma della curva e' parte del commento di ciascun modello richiesto
    dalla consegna, e va conservata come artefatto invece di essere ricalcolata
    a posteriori.

Controllo sui bordi
    Se la configurazione selezionata cade su un estremo di un parametro
    ordinato, il minimo potrebbe trovarsi fuori dalla griglia. Il controllo e'
    eseguito e registrato sempre, non solo quando il risultato sembra
    sospetto: e' la regola fissata prima di vedere i numeri.

Convergenza
    I modelli stimati per discesa coordinata possono non convergere entro il
    numero massimo di iterazioni su alcune configurazioni della griglia. Le
    mancate convergenze sono contate e riportate: una configurazione non
    convergente produce un punteggio che non e' confrontabile con gli altri, e
    ignorare l'avviso significherebbe lasciarla entrare in graduatoria senza
    che nulla lo segnali.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from src.protocol import as_cv

# Metriche calcolate su ogni configurazione della griglia. La selezione avviene
# sulla prima; le altre due sono riportate e non partecipano alla scelta.
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
REFIT_ON = "rmse"


@dataclass
class SearchResult:
    """Esito completo di una ricerca su griglia."""

    best_params: dict
    best_score: float
    table: pd.DataFrame = field(repr=False)
    boundary: list[str] = field(default_factory=list)
    convergence_warnings: int = 0
    seconds: float = 0.0

    @property
    def on_boundary(self) -> bool:
        return bool(self.boundary)


def _boundary_parameters(param_grid: dict, best_params: dict) -> list[str]:
    """Parametri la cui configurazione selezionata cade su un estremo della griglia.

    Il controllo riguarda i soli parametri con almeno tre valori ordinabili:
    su una griglia di due valori l'estremo e' inevitabile e l'avviso sarebbe
    privo di contenuto.
    """
    flagged = []
    for name, values in param_grid.items():
        values = list(values)
        if len(values) < 3:
            continue
        try:
            ordered = sorted(values)
        except TypeError:
            continue
        chosen = best_params.get(name)
        if chosen is None:
            continue
        if chosen == ordered[0]:
            flagged.append(f"{name}=minimo")
        elif chosen == ordered[-1]:
            flagged.append(f"{name}=massimo")
    return flagged


def grid_search(
    estimator,
    param_grid: dict,
    X,
    y,
    splits,
    *,
    n_jobs: int = -1,
) -> SearchResult:
    """Valuta la griglia sulle partizioni indicate e seleziona sulla metrica di riferimento.

    Le partizioni sono passate esplicitamente e non ricostruite qui: sono le
    stesse strutture usate da `src.protocol.evaluate`, quindi la ricerca e la
    rivalutazione avvengono sotto lo stesso vincolo di gruppo.
    """
    import time

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=as_cv(splits),
            scoring=SCORING,
            refit=REFIT_ON,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        search.fit(X, np.asarray(y))
    seconds = time.perf_counter() - start

    n_convergence = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    table = pd.DataFrame(search.cv_results_)
    keep = [c for c in table.columns if c.startswith("param_")]
    keep += [f"mean_test_{m}" for m in SCORING] + [f"std_test_{m}" for m in SCORING]
    keep += ["mean_fit_time", "rank_test_rmse"]
    table = table[keep].copy()

    # I punteggi di scikit-learn sono orientati in modo che valori maggiori
    # siano migliori: errore e errore assoluto sono percio' restituiti negati e
    # vengono riportati alla loro scala naturale.
    for metric in ("rmse", "mae"):
        table[f"mean_test_{metric}"] = -table[f"mean_test_{metric}"]
    table = table.sort_values("rank_test_rmse").reset_index(drop=True)

    return SearchResult(
        best_params=dict(search.best_params_),
        best_score=float(-search.best_score_),
        table=table,
        boundary=_boundary_parameters(param_grid, search.best_params_),
        convergence_warnings=n_convergence,
        seconds=seconds,
    )