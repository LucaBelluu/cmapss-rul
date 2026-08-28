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

Griglie del blocco che supera la linearita'
    Le quattro tecniche del laboratorio 8 governano tutte la stessa quantita',
    la flessibilita' concessa alla relazione fra letture e vita residua, con
    parametri diversi: il grado del polinomio, il numero di intervalli, il
    numero di nodi e il grado della base, l'intensita' della penalizzazione
    sulla curvatura. Le griglie sono fissate in modo che ciascuna copra un
    intervallo che va da un modello quasi lineare a uno chiaramente
    sovraparametrizzato, cosi' che la cross-validation abbia da entrambi i lati
    lo spazio per individuare un minimo interno.

    I costi sono stati misurati prima di fissare le griglie, su matrici della
    forma di quelle del progetto: nessuna configurazione supera i quattro
    secondi per addestramento, e il blocco intero resta nell'ordine dei dieci
    minuti per sottoinsieme. L'unica configurazione costosa e' il grado 4 del
    polinomio, che genera 7.314 colonne e richiede circa 140 secondi per
    addestramento oltre a 772 MB per la sola matrice espansa: non e' in griglia
    e vi entrerebbe solo se la regola sui bordi lo imponesse, nel qual caso
    l'esecuzione va lanciata riducendo il numero di processi paralleli.

    La collocazione dei punti di taglio delle step functions e dei nodi delle
    spline e' essa stessa un iperparametro, con i due valori corrispondenti a
    tagli equispaziati sull'escursione della variabile e a tagli sui quantili.
    Il controllo sui bordi non si applica a un parametro con due soli valori,
    ed e' corretto che non si applichi: su un parametro non ordinato la
    posizione estrema non ha significato.

Griglie del blocco della famiglia ad albero
    I sei modelli dei laboratori 9 e 10 governano l'errore per due vie diverse:
    l'aggregazione di alberi cresciuti per intero riduce la varianza della
    stima, l'addizione di alberi poco profondi ne riduce la distorsione. Le
    griglie sono fissate perche' la tabella renda leggibile questa differenza
    invece di sovrapporla a differenze di budget di ricerca.

    Potatura per cost-complexity. Il laboratorio ricava la sequenza dei valori
    di potatura dai dati, con `cost_complexity_pruning_path`, e sceglie il
    valore che minimizza l'errore sull'insieme di verifica. Nessuna delle due
    cose e' trasferibile: la scelta guarderebbe i dati su cui si misura il
    risultato, e una sequenza ricavata dai dati cambia da fold a fold, quindi
    non definisce una griglia comune alle partizioni ne' ai due sottoinsiemi.
    La griglia e' percio' un insieme di valori fissato a priori, uguale ovunque.
    La sua scala non e' arbitraria: il parametro e' nelle unita' dell'impurita',
    cioe' cicli al quadrato, e l'impurita' della radice e' la varianza del
    target, circa 1.700 su entrambi i sottoinsiemi. Un valore superiore a
    quell'ordine pota fino alla radice, uno prossimo a zero non pota, e
    l'intervallo copre quindi per costruzione l'intero percorso di potatura.
    Lo zero e' incluso ed e' l'albero cresciuto per intero.

    Conseguenza da tenere presente nella lettura: fra due valori consecutivi
    della sequenza propria dei dati la potatura non cambia, quindi piu' punti
    della griglia producono lo stesso albero. Il numero di alberi distinti che
    la griglia realizza e' minore del numero di configurazioni valutate, ed e'
    la quantita' che dice se la risoluzione della griglia sia sufficiente.

    Numero di alberi degli insiemi per aggregazione. Non e' un iperparametro:
    l'errore decresce in valore atteso in modo monotono nel numero di alberi e
    satura, quindi non governa un compromesso ma la precisione di una media.
    Metterlo in griglia farebbe selezionare sistematicamente il valore massimo
    e chiederebbe alla regola sui bordi un'estensione senza fine. E' fissato a
    300 per bagging e foresta, sopra i valori del laboratorio, e la scelta e'
    verificata dalla curva di saturazione misurata prima dell'esecuzione: fra
    300 e 500 alberi l'errore si muove di meno di 0,03 cicli su entrambi i
    sottoinsiemi, contro una dispersione fra fold di 1,1 cicli, e su FD003 non
    e' nemmeno monotono, perche' oltre le poche centinaia di alberi la
    variazione residua e' rumore della partizione.

    Bagging. Con alberi cresciuti per intero e numero di alberi fissato non ha
    iperparametri, ed entra in tabella senza configurazione come la regressione
    lineare multipla. E' lo stesso modello della foresta quando ogni divisione
    puo' scegliere fra tutte le variabili: la coincidenza dei due errori a
    parita' di numero di alberi e' un controllo di correttezza, come lo e' la
    coincidenza fra regressione sulle componenti principali a componenti
    complete e minimi quadrati.

    Foresta casuale. La frazione di variabili candidate a ciascuna divisione e'
    espressa come frazione e non come conteggio, perche' i due sottoinsiemi
    hanno 18 e 19 colonne e la griglia deve restare letteralmente la stessa. La
    dimensione minima della foglia e' il secondo asse: e' l'unico modo di
    ridurre la crescita degli alberi che il laboratorio impieghi, e governa
    anche la memoria occupata dall'insieme, che con alberi non potati arriva a
    quasi quattro milioni di nodi.

    Gradient boosting e sua implementazione esterna ricevono griglie identiche,
    sugli stessi tre assi e con gli stessi valori. Il confronto fra le due righe
    riguarda cosi' l'implementazione e non il budget di ricerca; la differenza
    che resta e' la regolarizzazione esplicita che l'implementazione esterna
    applica per impostazione predefinita, che non viene azzerata e va dichiarata
    nella lettura. AdaBoost ha gli stessi tre assi ma valori propri, perche' il
    suo tasso di apprendimento pesa il contributo di ciascuno stadio in modo
    diverso e i valori piccoli del gradient boosting vi corrisponderebbero a un
    modello che non ha il tempo di formarsi.

    Costi misurati prima di fissare le griglie, sulla prima partizione del seme
    di ricerca: la configurazione piu' costosa e' il gradient boosting a 600
    stadi e profondita' 5, con 25,6 s per addestramento su FD001 e 31,6 su
    FD003, e la griglia dei modelli per addizione somma circa 250 s per fold.
    L'implementazione esterna esegue la stessa configurazione in mezzo secondo.
    La ricerca sulla foresta va lanciata riducendo il numero di processi
    paralleli se la memoria e' scarsa: un insieme di 300 alberi non potati
    occupa circa 240 MB e la ricerca ne tiene in vita una copia per processo.

    Estensioni applicate. Le griglie dei tre modelli per addizione non sono
    quelle di partenza: le prime esecuzioni hanno selezionato configurazioni su
    bordi, e la regola le ha estese dal lato toccato, con un punto per asse,
    mantenendo la spaziatura propria dell'asse e sempre su entrambi i
    sottoinsiemi, perche' una griglia diversa fra i due renderebbe le due
    repliche non piu' condotte sotto lo stesso protocollo. I valori di partenza
    restano dentro la griglia: toglierli perche' hanno ottenuto punteggi
    peggiori sarebbe una selezione a posteriori sulla griglia stessa.

    Gradient boosting e la sua implementazione esterna avevano selezionato la
    profondita' massima su FD003. Esteso l'asse a 8, entrambi confermano la
    configurazione precedente su entrambi i sottoinsiemi: il bordo non
    vincolava, e la colonna aggiunta resta in griglia con i suoi punteggi.

    AdaBoost ha selezionato il tasso minimo, la profondita' massima e, dopo la
    prima estensione, anche il numero massimo di stadi. Gli assi del numero di
    stadi e della profondita' sono estesi a 800 e a 6. L'asse del tasso non
    viene esteso, per la ragione descritta sotto.

Il bordo inferiore del tasso di apprendimento di AdaBoost
    Nell'implementazione di AdaBoost.R2 il peso di ciascuno stadio e' il tasso
    di apprendimento moltiplicato per il logaritmo dell'inverso dell'errore
    relativo, quindi il tasso riscala tutti i pesi della stessa costante.
    L'aggregazione e' una mediana pesata, che individua lo stadio in cui la
    somma cumulata dei pesi supera meta' del totale ed e' percio' invariante a
    un riscalamento comune. Il tasso non agisce dunque sulla predizione
    attraverso i pesi degli stadi, ma soltanto attraverso l'aggiornamento dei
    pesi delle osservazioni, che e' l'errore relativo elevato a una potenza
    proporzionale al tasso.

    Quando il tasso tende a zero il ripesaggio si annulla: ogni stadio viene
    adattato su un campione bootstrap a pesi uniformi e le predizioni sono
    combinate per mediana. Quel limite e' il bagging, che il confronto contiene
    gia' con una riga propria. Il bordo inferiore di questo asse e' quindi
    strutturale nello stesso senso della potatura nulla e della frazione
    unitaria di variabili candidate: sotto non c'e' un modello nuovo, c'e' un
    modello gia' in tabella. L'estensione non apre spazio e non viene eseguita.

    La proprieta' riguarda il solo asse del tasso. Sull'asse della profondita'
    l'estensione resta dovuta, perche' con un tasso interno alla griglia il
    modello non degenera e alberi di base piu' profondi sono un modello diverso,
    non un modello gia' presente.

Chiusura della catena di estensioni
    Un'estensione che sposta il modello di meno della dispersione fra fold ha
    raggiunto la regione in cui il protocollo non distingue: la regola di
    lettura del progetto non ordina due risultati a quella distanza, quindi
    continuare a estendere inseguirebbe differenze che il progetto stesso
    dichiara illeggibili. La catena si chiude quando questo accade su entrambi i
    sottoinsiemi, e il limite viene dichiarato nella lettura del modello. Il
    criterio vale come condizione di arresto di una catena gia' iniziata, non
    come motivo per non applicare la regola sui bordi.

    Regola sui bordi, fissata prima dell'esecuzione. Sono bordi strutturali, che
    non producono estensione perche' oltre non esiste modello, il valore nullo
    della potatura (l'albero non potato), la frazione unitaria di variabili
    candidate (tutte le variabili, cioe' il bagging, che e' gia' in tabella come
    modello a se') e la foglia minima di una osservazione. Sono bordi veri, che
    producono estensione se selezionati, il valore massimo della potatura, la
    frazione minima di variabili, la foglia minima massima, e tutti e tre gli
    assi dei modelli per addizione. Il controllo del codice segnala entrambe le
    situazioni senza distinguerle: la distinzione e' questa, ed e' fissata qui
    prima di vedere i risultati.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.nonlinear import (
    GamRegressor,
    expansion_terms,
    gam_terms,
    polynomial_estimator,
    spline_estimator,
    step_estimator,
)
from src.trees import impurity_importances

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

# Griglie del blocco che supera la linearita'. Il grado 1 del polinomio e'
# incluso perche' e' il caso in cui l'espansione non aggiunge nulla: se viene
# selezionato, il modello coincide con la regressione lineare multipla, e la
# coincidenza e' una risposta sulla forma della relazione.
#
# Tre griglie sono state estese in applicazione della regola sui bordi, dopo
# che la prima esecuzione aveva selezionato una configurazione estrema: il
# numero di intervalli delle step functions su FD003, il numero di funzioni di
# base del modello additivo su entrambi i sottoinsiemi, e il grado della base
# spline verso il basso su FD003. L'estensione e' applicata a entrambi i
# sottoinsiemi e non al solo sottoinsieme in cui il bordo e' stato toccato:
# griglie diverse sui due sottoinsiemi renderebbero le due repliche del
# confronto non piu' condotte sotto lo stesso protocollo.
#
# La griglia del grado della base spline non e' invece estesa verso il basso,
# per due ragioni indipendenti. Il grado 0 produce funzioni indicatrici su
# intervalli, cioe' la stessa costruzione delle step functions con tagli di
# ampiezza uguale: il limite inferiore e' quindi un modello gia' presente in
# tabella, come accade a Elastic Net verso il lato di Ridge, e l'estensione non
# aprirebbe uno spazio nuovo. In piu' quella configurazione non e' valutabile:
# con grado 0 e estrapolazione costante la trasformazione fallisce su qualunque
# valore fuori dall'intervallo osservato in addestramento, condizione che si
# verifica in ogni fold. Il difetto e' circoscritto a quella combinazione, ed e'
# stato verificato che tutte le altre combinazioni di grado ed estrapolazione
# funzionano.
#
# L'estrapolazione resta quella predefinita, che mantiene costante la base fuori
# dall'intervallo osservato. L'alternativa che proseguirebbe l'andamento
# polinomiale produce fuori intervallo valori di base di ampiezza crescente,
# quindi predizioni instabili proprio sulle unita' che il modello non ha visto.
POLY_DEGREES = [1, 2, 3]
STEP_N_BINS = [3, 5, 8, 12, 20, 30, 50, 80]
STEP_STRATEGIES = ["uniform", "quantile"]
SPLINE_N_KNOTS = [3, 5, 8, 12, 20]
SPLINE_DEGREES = [1, 2, 3]
SPLINE_KNOTS = ["uniform", "quantile"]
GAM_LAMS = np.logspace(-3, 5, 9)
GAM_N_SPLINES = [5, 10, 20, 30, 40]

# Griglie del blocco della famiglia ad albero.
#
# Seme degli stimatori che ne richiedono uno. E' distinto dai semi del
# protocollo, che governano il partizionamento: questo riguarda il
# campionamento bootstrap e la scelta delle variabili candidate a ciascuna
# divisione, non quali motori finiscono da che parte.
TREE_SEED = 0

# Numero di alberi degli insiemi per aggregazione, fissato e non cercato.
N_TREES = 300

# La potatura parte dall'albero non potato e arriva oltre l'impurita' della
# radice, che vale circa 1.700 su entrambi i sottoinsiemi: la griglia copre
# percio' l'intero percorso, dal nessun taglio all'albero ridotto alla sola
# radice.
CCP_ALPHAS = np.concatenate([[0.0], np.logspace(-2, 3, 26)])

FOREST_MAX_FEATURES = [0.2, 0.33, 0.5, 0.7, 1.0]
FOREST_MIN_SAMPLES_LEAF = [1, 5, 20]

ADABOOST_N_ESTIMATORS = [50, 100, 200, 400, 800]
ADABOOST_LEARNING_RATES = [0.01, 0.05, 0.2, 1.0]
ADABOOST_MAX_DEPTHS = [2, 3, 4, 5, 6]

# Griglia condivisa fra le due implementazioni di gradient boosting.
BOOSTING_N_ESTIMATORS = [100, 300, 600]
BOOSTING_LEARNING_RATES = [0.01, 0.05, 0.1]
BOOSTING_MAX_DEPTHS = [2, 3, 5, 8]


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


NONLINEAR_MODELS: dict[str, ModelSpec] = {
    "polynomial": ModelSpec(
        key="polynomial",
        label="Regressione polinomiale",
        estimator=polynomial_estimator(),
        grid={"model__expand__degree": POLY_DEGREES},
        reader=expansion_terms,
        note="espansione con interazioni: e' l'unico modello del blocco non additivo",
    ),
    "step_functions": ModelSpec(
        key="step_functions",
        label="Step functions",
        estimator=step_estimator(),
        grid={
            "model__expand__n_bins": STEP_N_BINS,
            "model__expand__strategy": STEP_STRATEGIES,
        },
        reader=expansion_terms,
        note="collocazione dei tagli cercata in griglia fra ampiezza uguale e quantili",
    ),
    "spline": ModelSpec(
        key="spline",
        label="Regression spline",
        estimator=spline_estimator(),
        grid={
            "model__expand__n_knots": SPLINE_N_KNOTS,
            "model__expand__degree": SPLINE_DEGREES,
            "model__expand__knots": SPLINE_KNOTS,
        },
        reader=expansion_terms,
        note="base B-spline su ciascuna variabile, numero di nodi e grado in griglia",
    ),
    "gam": ModelSpec(
        key="gam",
        label="Modello additivo generalizzato",
        estimator=GamRegressor(),
        grid={"model__lam": GAM_LAMS, "model__n_splines": GAM_N_SPLINES},
        reader=gam_terms,
        note="un termine liscio per variabile, penalizzazione comune a tutti i termini",
    ),
}

def xgboost_estimator() -> XGBRegressor:
    """Regressore dell'implementazione esterna di gradient boosting.

    Riceve un solo processo perche' dentro la ricerca su griglia il parallelismo
    e' gia' speso sulle configurazioni: lasciarlo occupare tutti i processori
    mentre gli altri modelli ne usano uno renderebbe non confrontabili i tempi
    riportati in tabella.

    Il tipo di importanza e' fissato al guadagno complessivo. Per impostazione
    predefinita la libreria restituisce il guadagno medio per divisione, che non
    e' la stessa quantita' riportata da scikit-learn: sotto lo stesso nome la
    tabella conterrebbe due grandezze diverse a seconda della riga.
    """
    return XGBRegressor(
        objective="reg:squarederror",
        importance_type="total_gain",
        random_state=TREE_SEED,
        n_jobs=1,
    )


TREE_MODELS: dict[str, ModelSpec] = {
    "tree": ModelSpec(
        key="tree",
        label="Albero di regressione potato",
        estimator=DecisionTreeRegressor(random_state=TREE_SEED),
        grid={"model__ccp_alpha": CCP_ALPHAS},
        reader=impurity_importances,
        note="potatura per cost-complexity su griglia fissata a priori, non sulla "
        "sequenza ricavata dai dati",
    ),
    "bagging": ModelSpec(
        key="bagging",
        label="Bagging di alberi",
        estimator=BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=TREE_SEED),
            n_estimators=N_TREES,
            bootstrap=True,
            random_state=TREE_SEED,
        ),
        reader=impurity_importances,
        note="alberi non potati, numero di alberi fissato e non cercato: nessun "
        "iperparametro da selezionare",
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Foresta casuale",
        estimator=RandomForestRegressor(n_estimators=N_TREES, random_state=TREE_SEED),
        grid={
            "model__max_features": FOREST_MAX_FEATURES,
            "model__min_samples_leaf": FOREST_MIN_SAMPLES_LEAF,
        },
        reader=impurity_importances,
        note="la frazione unitaria di variabili candidate e' il bagging, gia' in "
        "tabella come modello a se'",
    ),
    "adaboost": ModelSpec(
        key="adaboost",
        label="AdaBoost",
        estimator=AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=TREE_SEED),
            random_state=TREE_SEED,
        ),
        grid={
            "model__n_estimators": ADABOOST_N_ESTIMATORS,
            "model__learning_rate": ADABOOST_LEARNING_RATES,
            "model__estimator__max_depth": ADABOOST_MAX_DEPTHS,
        },
        reader=impurity_importances,
        note="profondita' dell'albero di base in griglia insieme al numero di stadi "
        "e al tasso di apprendimento",
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting",
        label="Gradient boosting",
        estimator=GradientBoostingRegressor(random_state=TREE_SEED),
        grid={
            "model__n_estimators": BOOSTING_N_ESTIMATORS,
            "model__learning_rate": BOOSTING_LEARNING_RATES,
            "model__max_depth": BOOSTING_MAX_DEPTHS,
        },
        reader=impurity_importances,
        note="griglia condivisa con l'implementazione esterna, sugli stessi assi e "
        "sugli stessi valori",
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        label="XGBoost",
        estimator=xgboost_estimator(),
        grid={
            "model__n_estimators": BOOSTING_N_ESTIMATORS,
            "model__learning_rate": BOOSTING_LEARNING_RATES,
            "model__max_depth": BOOSTING_MAX_DEPTHS,
        },
        reader=impurity_importances,
        note="stessa griglia del gradient boosting di scikit-learn, con la "
        "regolarizzazione predefinita della libreria non azzerata",
    ),
}