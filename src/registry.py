"""Registro dei modelli del confronto e delle rispettive griglie di iperparametri.

Ruolo nel progetto
    Raccoglie in un unico punto, per ciascun modello, lo stimatore, la griglia
    su cui viene cercata la sua configurazione e la funzione che ne estrae i
    parametri leggibili. Ogni blocco del confronto aggiunge qui il proprio
    registro: e' cosi' che il motore di esperimento resta identico per tutti i
    blocchi e nessun modello puo' ricevere un trattamento diverso dagli altri.

Cosa riceve
    Nulla dal chiamante, se non il numero di variabili della matrice per le
    griglie che ne dipendono.

Cosa produce
    Strutture `ModelSpec` pronte per `src.experiment`.

Criterio con cui sono fissate le griglie
    Le griglie del laboratorio 7 sono tarate su una matrice di 331 righe e 10
    variabili. Trasportarle senza controllo su 16.500 righe di addestramento
    per fold non e' automatico, perche' scikit-learn parametrizza in modo
    diverso le due penalita': Ridge minimizza la somma dei quadrati dei
    residui piu' la penalita', mentre Lasso ed Elastic Net dividono la parte di
    errore per il numero di righe. A parita' di valore numerico del parametro,
    la contrazione prodotta su Ridge e' quindi piu' debole di un fattore pari
    al numero di righe. La griglia di Ridge e' estesa verso l'alto per questo
    motivo, quella di Lasso e' lasciata nella forma del laboratorio.

    A ogni ricerca si applica una regola fissata prima di vedere i risultati:
    se la configurazione selezionata cade su un estremo della griglia, la
    griglia non e' un intervallo dentro cui si trova un minimo ma un vincolo
    che lo taglia fuori, e va estesa da quel lato e la ricerca rieseguita. La
    regola e' verificata dal codice in `src.search` e il suo esito e'
    registrato negli artefatti dell'esperimento: e' cio' che distingue
    l'estensione motivata di una griglia dalla ricerca a posteriori del numero
    migliore.

    Elastic Net cerca anche sul bilanciamento fra le due penalita'. Lasciarlo
    al valore predefinito, come nel laboratorio, ridurrebbe il modello a una
    via di mezzo fissata a priori fra Ridge e Lasso, e la riga corrispondente
    della tabella non rappresenterebbe la tecnica.

    La griglia del bilanciamento e' infittita a entrambi gli estremi. Verso il
    lato della penalita' di tipo Lasso perche' e' li' che il comportamento del
    modello cambia piu' rapidamente. Verso il lato opposto perche' il limite
    inferiore del bilanciamento e' Ridge: se la selezione cade sul valore piu'
    piccolo della griglia, la lettura corretta e' che il dato chieda una
    penalita' di tipo Ridge, ed e' quindi necessario che i valori piccoli siano
    rappresentati abbastanza fittamente da rendere leggibile l'avvicinamento.
    Il valore nullo non e' incluso: coincide con Ridge, che compare in tabella
    come modello a se', e la sua stima per discesa coordinata non e' quella
    usata dal modello dedicato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

# Numero massimo di iterazioni per i modelli stimati per discesa coordinata.
# Il laboratorio usa 10.000 su 331 righe; il valore e' alzato perche' qui le
# righe sono due ordini di grandezza di piu' e la mancata convergenza di una
# sola configurazione della griglia produrrebbe un punteggio non confrontabile
# con gli altri. Le mancate convergenze residue sono contate e registrate
# dagli esperimenti anziche' soppresse.
MAX_ITER = 50_000

# La griglia di Ridge copre un intervallo piu' ampio di quella del laboratorio
# per la ragione di scala descritta sopra. L'ampiezza si e' rivelata non
# necessaria (la configurazione selezionata cade in una zona coperta anche
# dalla griglia del laboratorio) ed e' mantenuta perche' il suo costo e' nullo
# e perche' documenta che la selezione non e' vincolata dall'estremo.
RIDGE_ALPHAS = np.logspace(-2, 8, 41)
LASSO_ALPHAS = np.logspace(-4, 4, 50)
ENET_ALPHAS = np.logspace(-4, 4, 50)
ENET_L1_RATIOS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


@dataclass(frozen=True)
class ModelSpec:
    """Un modello del confronto, con tutto cio' che serve a valutarlo.

    key
        Identificativo usato nei nomi dei file e nelle tabelle.
    label
        Nome esteso per le tabelle destinate alla lettura.
    estimator
        Stimatore nudo, senza pre-processing: la standardizzazione viene
        aggiunta dal motore di esperimento, uguale per tutti i modelli.
    grid
        Griglia degli iperparametri, con le chiavi nella forma attesa dalla
        pipeline. Puo' essere una funzione del numero di variabili per le
        griglie che ne dipendono. Vuota per i modelli senza iperparametri.
    reader
        Funzione che estrae dalla pipeline adattata i parametri leggibili del
        modello, in forma tabellare. E' il materiale con cui si costruisce il
        commento richiesto dalla consegna.
    note
        Annotazione sul modello, riportata negli artefatti.
    """

    key: str
    label: str
    estimator: BaseEstimator
    grid: dict | Callable[[int], dict] = field(default_factory=dict)
    reader: Callable[[Pipeline, list[str]], pd.DataFrame] | None = None
    note: str = ""

    def param_grid(self, n_features: int) -> dict:
        return self.grid(n_features) if callable(self.grid) else dict(self.grid)


def linear_coefficients(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Coefficienti di un modello lineare, nella scala delle variabili standardizzate.

    Le variabili entrano nel modello dopo standardizzazione, quindi i
    coefficienti sono confrontabili fra loro in ampiezza: e' la lettura con cui
    si commenta quali variabili il modello usa e quali annulla.
    """
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_).ravel()
    frame = pd.DataFrame({"feature": feature_names, "coef": coef})
    frame["abs_coef"] = frame["coef"].abs()
    frame["zero"] = np.isclose(frame["coef"], 0.0)
    return frame.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def pcr_coefficients(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Coefficienti della regressione sulle componenti principali, riproiettati.

    Il modello stima i coefficienti nello spazio delle componenti, dove non
    sono direttamente commentabili. Riproiettandoli sulle variabili originali
    si ottiene il coefficiente complessivo di ciascuna variabile, che rende
    confrontabile questo modello con gli altri modelli lineari. La quota di
    varianza spiegata dalle componenti trattenute e' riportata a parte, perche'
    e' la quantita' che descrive quanto della matrice il modello conserva.
    """
    inner = pipeline.named_steps["model"]
    pca = inner.named_steps["pca"]
    linreg = inner.named_steps["linreg"]

    coef = np.asarray(linreg.coef_).ravel() @ pca.components_
    frame = pd.DataFrame({"feature": feature_names, "coef": coef})
    frame["abs_coef"] = frame["coef"].abs()
    frame["zero"] = np.isclose(frame["coef"], 0.0)
    frame["n_components"] = pca.n_components_
    frame["explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())
    return frame.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def pcr_estimator() -> Pipeline:
    """Riduzione a componenti principali seguita da regressione lineare.

    Le componenti sono calcolate dentro la pipeline, quindi su ciascuna parte
    di addestramento e mai sull'intera matrice: la riduzione della
    dimensionalita' e' pre-processing e sta dentro il flusso di validazione
    come la standardizzazione.
    """
    return Pipeline([("pca", PCA()), ("linreg", LinearRegression())])


LINEAR_MODELS: dict[str, ModelSpec] = {
    "ols": ModelSpec(
        key="ols",
        label="Regressione lineare multipla",
        estimator=LinearRegression(),
        reader=linear_coefficients,
        note="nessun iperparametro, valutata direttamente sulle partizioni di confronto",
    ),
    "ridge": ModelSpec(
        key="ridge",
        label="Ridge",
        estimator=Ridge(),
        grid={"model__alpha": RIDGE_ALPHAS},
        reader=linear_coefficients,
        note="griglia estesa verso l'alto rispetto al laboratorio per la diversa "
        "parametrizzazione della penalita' rispetto a Lasso",
    ),
    "lasso": ModelSpec(
        key="lasso",
        label="Lasso",
        estimator=Lasso(max_iter=MAX_ITER),
        grid={"model__alpha": LASSO_ALPHAS},
        reader=linear_coefficients,
        note="griglia del laboratorio",
    ),
    "elastic_net": ModelSpec(
        key="elastic_net",
        label="Elastic Net",
        estimator=ElasticNet(max_iter=MAX_ITER),
        grid={"model__alpha": ENET_ALPHAS, "model__l1_ratio": ENET_L1_RATIOS},
        reader=linear_coefficients,
        note="ricerca anche sul bilanciamento fra le due penalita'",
    ),
    "pcr": ModelSpec(
        key="pcr",
        label="Regressione sulle componenti principali",
        estimator=pcr_estimator(),
        grid=lambda n_features: {"model__pca__n_components": list(range(1, n_features + 1))},
        reader=pcr_coefficients,
        note="griglia completa da una componente al numero di variabili",
    ),
}

# Modelli del blocco che passano dalla selezione delle variabili e non da una
# griglia di iperparametri. Sono elencati qui perche' il blocco sia descritto
# in un unico punto, ma il loro percorso di ricerca e' in `src.selection`.
SELECTION_MODELS: dict[str, str] = {
    "best_subset": "Best subset selection",
    "forward_stepwise": "Forward stepwise selection",
    "backward_stepwise": "Backward stepwise selection",
}