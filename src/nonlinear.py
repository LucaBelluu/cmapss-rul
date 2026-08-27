"""Modelli del superamento della linearita': espansioni di base e modello additivo.

Ruolo nel progetto
    Fornisce gli stimatori e le funzioni di lettura del blocco che porta in
    confronto le tecniche del laboratorio 8 (regressione polinomiale, step
    functions, regression spline, modello additivo generalizzato). Gli
    stimatori sono composti in `src.registry` e valutati dal motore di
    esperimento, che resta identico a quello degli altri blocchi.

Cosa riceve
    Nulla dal chiamante: le funzioni costruiscono stimatori non adattati. Le
    funzioni di lettura ricevono una pipeline gia' adattata e l'elenco dei nomi
    delle variabili originali.

Cosa produce
    Stimatori conformi all'interfaccia di scikit-learn e tabelle dei parametri
    leggibili di ciascun modello.

Perche' le tecniche sono applicate a tutte le variabili
    Il laboratorio le presenta su una variabile per volta a scopo didattico,
    perche' la curva stimata sia disegnabile. Un modello costruito su una sola
    variabile non sarebbe confrontabile con quelli degli altri blocchi, che
    usano l'intera matrice: qui l'espansione e' applicata a ogni variabile e i
    termini risultanti entrano insieme in una regressione lineare, che e' la
    forma in cui il laboratorio stesso usa `SplineTransformer` quando passa a
    piu' variabili.

Collocazione dei punti di taglio
    Per le step functions e per le spline la posizione dei tagli e' un
    iperparametro con due valori: intervalli di ampiezza uguale sull'escursione
    della variabile, che e' la costruzione del laboratorio, oppure tagli sui
    quantili. La differenza non e' cosmetica: le letture dei sensori non sono
    distribuite uniformemente sul proprio intervallo, e con tagli equispaziati
    gli intervalli estremi possono contenere pochissime osservazioni, cosi' che
    il coefficiente corrispondente viene stimato su una manciata di righe. I
    tagli sui quantili distribuiscono le osservazioni in parti uguali. Lasciare
    scegliere alla cross-validation trasforma una scelta arbitraria in una
    misura, e il costo aggiuntivo e' di pochi secondi.

Adattatore per il modello additivo
    `pygam.LinearGAM` non implementa `__sklearn_tags__`, introdotto come
    requisito dalla versione 1.6 di scikit-learn: la clonazione funziona, ma
    l'inserimento in una `Pipeline` fallisce in fase di adattamento. `GamRegressor`
    e' l'adattatore minimo che risolve il problema: eredita da `BaseEstimator`,
    espone gli iperparametri del modello come argomenti del costruttore e
    costruisce i termini dentro `fit`, quando il numero di colonne e' noto.
    Questo lo rende anche indipendente dal sottoinsieme, che ha 18 variabili su
    FD001 e 19 su FD003.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer

# Numero di punti su cui viene campionata ciascuna funzione parziale del
# modello additivo. E' una risoluzione di disegno: non influisce sulla stima.
PARTIAL_DEPENDENCE_POINTS = 100


class GamRegressor(BaseEstimator, RegressorMixin):
    """Modello additivo generalizzato con un termine liscio per ogni variabile.

    lam
        Intensita' della penalizzazione sulla curvatura, comune a tutti i
        termini. Valori grandi riportano ciascun termine verso una retta.
    n_splines
        Numero di funzioni di base per termine: fissa quanto ogni funzione puo'
        essere flessibile prima che la penalizzazione intervenga.
    spline_order
        Grado delle funzioni di base. Resta 3 e non entra in griglia: e' il
        valore del laboratorio, e la flessibilita' e' gia' governata dai due
        parametri precedenti.
    """

    def __init__(self, lam: float = 0.6, n_splines: int = 20, spline_order: int = 3, max_iter: int = 100):
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("GamRegressor richiede una matrice bidimensionale")

        terms = s(0, n_splines=self.n_splines, spline_order=self.spline_order)
        for j in range(1, X.shape[1]):
            terms = terms + s(j, n_splines=self.n_splines, spline_order=self.spline_order)

        self.n_features_in_ = X.shape[1]
        self.gam_ = LinearGAM(terms, lam=self.lam, max_iter=self.max_iter).fit(X, y)
        return self

    def predict(self, X):
        return self.gam_.predict(np.asarray(X, dtype=float))


def polynomial_estimator() -> Pipeline:
    """Espansione polinomiale di tutte le variabili seguita da regressione lineare.

    L'espansione comprende le interazioni, che e' la forma predefinita dello
    strumento usato nel laboratorio. E' anche cio' che distingue questo modello
    dagli altri del blocco: spline e modello additivo sono additivi per
    costruzione, quindi senza le interazioni il blocco non conterrebbe alcun
    modello capace di rappresentare l'effetto congiunto di due variabili.
    """
    return Pipeline(
        [
            ("expand", PolynomialFeatures(include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


class QuietKBinsDiscretizer(KBinsDiscretizer):
    """Discretizzazione in intervalli, senza l'avviso sugli intervalli degeneri.

    Su una variabile con pochi valori distinti, chiedere molti intervalli
    produce intervalli di ampiezza nulla, che lo strumento rimuove emettendo un
    avviso. La condizione e' attesa su questi dati (`sensor_10` assume quattro
    valori su FD003, e la censura del target rende quasi degenere la coda
    superiore di alcuni sensori), e l'esito e' corretto: la variabile riceve
    meno colonne, le altre non sono toccate.

    L'avviso viene percio' filtrato. Non e' una soppressione silenziosa: il
    numero di colonne effettivamente generate e' registrato negli artefatti
    dell'esperimento, ed e' da li' che si legge quanti intervalli sono stati
    rimossi. La differenza rispetto al lasciare l'avviso attivo e' che una
    ricerca su griglia lo emette una volta per configurazione e per fold,
    rendendo illeggibile il registro dell'esecuzione senza aggiungere
    informazione.

    Il filtro e' dentro `fit` e non a livello di modulo: uno scikit-learn che
    cambiasse il messaggio tornerebbe a mostrarlo, invece di nascondere un
    avviso diverso.
    """

    def fit(self, X, y=None, sample_weight=None):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Bins whose width are too small",
                category=UserWarning,
            )
            return super().fit(X, y, sample_weight=sample_weight)


def step_estimator() -> Pipeline:
    """Variabili indicatrici su intervalli di ciascuna variabile, poi regressione lineare.

    La discretizzazione e' adattata dentro la pipeline, quindi i punti di
    taglio sono calcolati sulla sola parte di addestramento di ogni fold.
    """
    return Pipeline(
        [
            ("expand", QuietKBinsDiscretizer(encode="onehot-dense")),
            ("linreg", LinearRegression()),
        ]
    )


def spline_estimator() -> Pipeline:
    """Base B-spline su ciascuna variabile seguita da regressione lineare.

    E' la regression spline del laboratorio: le funzioni di base sono generate
    dallo strumento e i coefficienti stimati per minimi quadrati, senza
    penalizzazione sulla curvatura. La flessibilita' e' governata dal numero di
    nodi e dal grado, che sono gli iperparametri cercati in griglia.
    """
    return Pipeline(
        [
            ("expand", SplineTransformer(include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def _source_variable(term: str, feature_names: list[str]) -> str:
    """Variabile originale da cui proviene un termine dell'espansione.

    I nomi prodotti dagli strumenti di scikit-learn antepongono la variabile di
    provenienza al proprio suffisso: `sensor_02^2` per una potenza,
    `sensor_02 sensor_03` per un prodotto, `sensor_02_sp_4` per una funzione di
    base, `sensor_02_1.0` per un intervallo. I fattori di un prodotto sono
    separati da uno spazio.

    L'attribuzione avviene per prefisso piu' lungo e non per contenimento:
    cercare il nome dentro il termine attribuirebbe a `sensor_01` anche i
    termini di `sensor_011`, se un nome fosse prefisso di un altro.

    Il termine che combina due variabili e' etichettato come interazione:
    attribuirlo a una delle due sarebbe arbitrario, e la quota di modello che
    sta nelle interazioni e' una delle letture del blocco.
    """
    sources = set()
    for factor in term.split(" "):
        matches = [name for name in feature_names if factor.startswith(name)]
        if matches:
            sources.add(max(matches, key=len))
    if not sources:
        return "sconosciuta"
    if len(sources) > 1:
        return "interazione"
    return sources.pop()


def expansion_terms(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Coefficienti dei termini dell'espansione, con la variabile di provenienza.

    I coefficienti dei termini di una base espansa non sono commentabili uno per
    uno, perche' descrivono l'effetto di una funzione di base e non di una
    variabile. La lettura utile e' aggregata: sommando le ampiezze dei termini
    che provengono dalla stessa variabile si ottiene quanto peso il modello
    attribuisce a ciascuna, in una forma confrontabile con i coefficienti dei
    modelli lineari. La tabella conserva il dettaglio per termine, cosi' che
    l'aggregazione avvenga nel notebook e sia rifacibile con criteri diversi.
    """
    inner = pipeline.named_steps["model"]
    expand = inner.named_steps["expand"]
    linreg = inner.named_steps["linreg"]

    terms = list(expand.get_feature_names_out(feature_names))
    coef = np.asarray(linreg.coef_).ravel()

    frame = pd.DataFrame({"feature": [_source_variable(t, feature_names) for t in terms], "term": terms, "coef": coef})
    frame["abs_coef"] = frame["coef"].abs()
    frame["zero"] = np.isclose(frame["coef"], 0.0)
    return frame.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def gam_terms(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Riepilogo per variabile del modello additivo.

    Per ciascun termine sono riportati i gradi di liberta' effettivi, che
    misurano quanta flessibilita' la penalizzazione ha lasciato alla funzione
    (un valore vicino a uno indica una funzione ormai indistinguibile da una
    retta), l'escursione della funzione parziale sul dominio della variabile,
    che dice di quanti cicli quella variabile sposta la predizione, e il livello
    di significativita' calcolato dalla libreria.

    L'escursione e' la quantita' confrontabile fra variabili, perche' e' nelle
    unita' del target. I gradi di liberta' effettivi ne descrivono la forma e
    non l'ampiezza: una variabile puo' avere una funzione molto curva e un
    effetto piccolo.
    """
    gam = pipeline.named_steps["model"].gam_
    stats = gam.statistics_
    edof_per_coef = np.asarray(stats["edof_per_coef"]).ravel()
    p_values = list(stats["p_values"])

    rows = []
    start = 0
    for i, name in enumerate(feature_names):
        term = gam.terms[i]
        n_coefs = term.n_coefs
        grid = gam.generate_X_grid(term=i, n=PARTIAL_DEPENDENCE_POINTS)
        effect = gam.partial_dependence(term=i, X=grid)
        rows.append(
            {
                "feature": name,
                "edof": float(edof_per_coef[start : start + n_coefs].sum()),
                "effect_range": float(np.max(effect) - np.min(effect)),
                "p_value": float(p_values[i]) if i < len(p_values) else np.nan,
                "lam": float(np.ravel(term.lam)[0]),
                "n_coefs": int(n_coefs),
            }
        )
        start += n_coefs

    frame = pd.DataFrame(rows)
    # La colonna `term` esiste anche qui, con il nome della variabile, perche'
    # le tabelle dei quattro modelli finiscono nello stesso artefatto e una
    # colonna presente per tre modelli su quattro andrebbe gestita a parte in
    # lettura.
    frame["term"] = frame["feature"]
    frame["coef"] = frame["effect_range"]
    frame["abs_coef"] = frame["effect_range"]
    frame["zero"] = False
    return frame.sort_values("effect_range", ascending=False).reset_index(drop=True)


def gam_partial_dependence(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Funzioni parziali del modello additivo, campionate su una griglia.

    E' la lettura con cui il laboratorio commenta il modello additivo: per ogni
    variabile, la forma dell'effetto sul target. La griglia e' generata dalla
    libreria sul dominio osservato della variabile.

    Le ascisse sono nella scala standardizzata, perche' la standardizzazione e'
    dentro la pipeline e il modello vede le variabili gia' trasformate. La
    conversione alla scala originale richiede media e scarto della parte di
    addestramento, che sono disponibili nello stadio di standardizzazione della
    stessa pipeline e vengono riportati in tabella per rendere il disegno
    leggibile in unita' fisiche.
    """
    gam = pipeline.named_steps["model"].gam_
    scaler = pipeline.named_steps["scale"]
    mean = np.asarray(scaler.mean_).ravel()
    scale = np.asarray(scaler.scale_).ravel()

    records = []
    for i, name in enumerate(feature_names):
        grid = gam.generate_X_grid(term=i, n=PARTIAL_DEPENDENCE_POINTS)
        effect, ci = gam.partial_dependence(term=i, X=grid, width=0.95)
        x_std = grid[:, i]
        records.append(
            pd.DataFrame(
                {
                    "feature": name,
                    "x_standardized": x_std,
                    "x_original": x_std * scale[i] + mean[i],
                    "effect": effect,
                    "ci_low": ci[:, 0],
                    "ci_high": ci[:, 1],
                }
            )
        )
    return pd.concat(records, ignore_index=True)