"""Misura dei costi della regressione a vettori di supporto e del percettrone
multistrato, preliminare al blocco dei metodi a margine e delle reti.

Ruolo nel progetto
    Precede l'esperimento del quarto blocco del confronto e non ne fa parte,
    come `scripts/measure_tree_costs.py` precede quello della famiglia ad
    albero. Le griglie del blocco sono fissate su una misura del costo e non su
    una stima, e la decisione su come trattare la regressione a vettori di
    supporto sotto il protocollo del progetto poggia sui numeri prodotti qui.
    Nessun risultato prodotto in questo script entra in graduatoria.

Perche' questa sonda e' piu' articolata di quella del blocco ad albero
    Il costo di adattamento della regressione a vettori di supporto con kernel
    cresce fra il quadrato e il cubo del numero di righe, e le righe di
    addestramento per fold sono 16.505 su FD001 e 19.776 su FD003. Un solo
    numero misurato a dimensione piena non basta a decidere: serve sapere come
    il costo cresce nelle righe, quale angolo della griglia lo fa esplodere e
    quanto pesa la dimensione della cache del kernel, perche' sono queste tre
    quantita' a distinguere le alternative fra cui la decisione va presa.

Cosa riceve
    I file grezzi in `data/raw/`, attraverso la catena `src.data`, `src.target`,
    `src.design`. Nessun argomento obbligatorio.

Cosa produce
    Un artefatto per sottoinsieme, riscritto conservando i blocchi non
    rieseguiti: un'esecuzione parziale sostituisce le righe dei soli blocchi che
    ha ricalcolato e lascia intatte le altre, cosi' che una misura lunga gia'
    fatta non vada persa lanciando un blocco aggiunto dopo.

    In `experiments/kernel_models/`, per ciascun sottoinsieme:

    - `{SUBSET}_kernel_cost_probe.csv`, una riga per misura, con il blocco di
      appartenenza, la configurazione, il numero di righe di addestramento, i
      tempi di adattamento e di predizione, il numero e la frazione di vettori
      di supporto, l'esito della misura e l'errore sulla partizione;
    - `{SUBSET}_mlp_cost_probe.csv`, la stessa struttura per il percettrone,
      con il numero di iterazioni effettivamente eseguite e l'indicazione di
      mancata convergenza;

    e, una volta sola, `environment.csv` con il numero di processori, la memoria
    e le versioni con cui le misure sono state prese, che sono le quantita' da
    cui la durata di un esperimento si ricava moltiplicando.

I quattro blocchi di misura
    `ladder` misura il tempo di adattamento di una configurazione centrale al
    crescere del numero di righe, su ciascuno dei tre kernel. E' il blocco da
    cui si legge l'esponente empirico di crescita e quindi se il modello sia
    praticabile a dimensione piena.

    `sensitivity` misura, a righe fissate, l'effetto di ciascun iperparametro
    lungo un asse per volta. Il costo di una griglia non e' il numero di
    configurazioni moltiplicato per un costo medio: e' dominato dall'angolo
    piu' oneroso, e questo blocco individua quale sia.

    `cache` confronta tre dimensioni della cache del kernel a righe fissate. La
    matrice del kernel a dimensione piena occuperebbe circa 2,2 GB su FD001,
    quindi il valore predefinito della libreria (200 MB) non la contiene e parte
    delle colonne viene ricalcolata a ogni passaggio. La dimensione della cache
    e' un parametro di implementazione e non del modello: cambiarla non altera
    la funzione stimata e non introduce quindi una differenza di condizioni fra
    modelli.

    `confirm` adatta una volta sola, a dimensione piena, l'angolo piu' oneroso
    fra quelli che la griglia conterra' plausibilmente. E' la misura che rende
    la decisione fondata su un numero osservato invece che su una estrapolazione;
    l'estrapolazione del blocco `ladder` serve soltanto a decidere se valga la
    pena tentarla.

    `mlp` misura il percettrone a dimensione piena su quattro configurazioni.

    `degree` misura il kernel polinomiale al variare del grado sull'angolo in cui
    la sua stima si degrada. Il prodotto interno fra due righe standardizzate a
    diciotto colonne e' dell'ordine delle diciotto unita', quindi il valore del
    kernel cresce come quel prodotto moltiplicato per `gamma` ed elevato al
    grado: con `gamma` alto e grado alto la matrice del kernel assume valori di
    ampiezza tale che l'ottimizzazione richiede milioni di iterazioni. Il blocco
    misura quanto costa quell'angolo con il tetto alle iterazioni attivo, prima
    che il grado entri o non entri in griglia.

    `curve` misura la traiettoria della perdita del percettrone al crescere
    delle iterazioni, con la stessa costruzione incrementale usata per la curva
    di saturazione degli insiemi: le iterazioni sono aggiunte a una rete gia'
    addestrata, quindi l'intera curva costa quanto il solo adattamento con il
    numero massimo di iterazioni. Serve a fissare il numero di iterazioni invece
    di cercarlo, per la stessa ragione per cui il numero di alberi e' fissato:
    non governa un compromesso ma la convergenza di una procedura.

    I blocchi successivi al primo non ripetono le misure che il primo ha gia'
    dichiarato fuori portata: un kernel che non termina entro il limite sulla
    configurazione centrale non termina nemmeno su una piu' onerosa alla stessa
    dimensione, e ritentarlo consumerebbe il limite di tempo una volta per
    configurazione senza aggiungere informazione. L'esclusione e' registrata
    riga per riga e non lasciata implicita nell'assenza della misura.

Sottocampionamento per diradamento
    I punti della scala non sono estratti a caso: da ciascun motore viene tenuto
    un ciclo ogni k, con k determinato dal numero di righe richiesto. Le righe di
    un motore sono cicli consecutivi della stessa traiettoria e sono quasi
    identiche fra loro, quindi il diradamento rimuove ripetizioni e mantiene
    tutti i motori e l'intera escursione del target, che un'estrazione casuale
    conserverebbe solo in media. E' anche la stessa operazione che il blocco
    potrebbe adottare come compromesso se il costo a dimensione piena risultasse
    proibitivo: misurarne qui il costo e l'errore la rende una quantita'
    osservata e non un'ipotesi.

Protezione contro le misure che non terminano
    Ogni adattamento e' eseguito in un processo separato con un limite di tempo.
    La libreria che stima il modello e' codice nativo che non restituisce il
    controllo all'interprete durante l'ottimizzazione, quindi un limite di tempo
    interno al processo non verrebbe applicato: una configurazione che non
    converge bloccherebbe la sonda a tempo indeterminato. Il superamento del
    limite non e' un errore da correggere ma un esito della misura, ed e'
    registrato come tale.

    La scala si interrompe anche in anticipo: prima di tentare un punto, il suo
    costo e' proiettato dai due punti gia' misurati, e se supera il budget il
    punto non viene tentato e la riga lo registra. La proiezione decide soltanto
    se tentare una misura, non sostituisce mai un valore misurato.

Errore riportato
    L'errore sulla partizione accompagna ogni misura come controllo di
    plausibilita' della catena e come lettura del costo del diradamento. Non e'
    un criterio con cui fissare gli estremi delle griglie: scegliere un
    intervallo perche' contiene il valore migliore osservato qui sarebbe una
    selezione fatta prima e fuori dal protocollo.

Come si lancia
    python -m scripts.measure_kernel_costs
    python -m scripts.measure_kernel_costs --quick
    python -m scripts.measure_kernel_costs --subsets FD001 --blocks ladder
    python -m scripts.measure_kernel_costs --timeout 1200 --budget 2400

    La modalita' `--quick` riduce la scala ai punti piccoli e i limiti di tempo,
    e serve a convalidare la catena prima di lanciare la misura vera.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import time
import warnings
from queue import Empty

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.design import SUBSETS_IN_SCOPE, build_design
from src.pipeline import build_pipeline
from src.protocol import N_SPLITS, SEARCH_SEEDS, make_splits, regression_metrics
from src.target import RUL_CAP

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "kernel_models"

# Seme degli stimatori che ne richiedono uno. E' distinto dai semi del
# protocollo, che governano il partizionamento: qui riguarda l'inizializzazione
# dei pesi della rete, non quali motori finiscono da che parte.
MODEL_SEED = 0

# Configurazione centrale della scala. I valori sono scelti al centro degli
# intervalli plausibili e non sono una griglia: servono a tenere fissa la
# configurazione mentre variano le righe.
#
# `epsilon` non e' trasferibile dal laboratorio, che lo usa a 0,1 su un target
# con deviazione standard di circa 1,15, cioe' a circa il 9 per cento della
# dispersione del target. Qui il target e' in cicli e ha deviazione standard di
# circa 41: lo stesso rapporto vale circa 4 cicli. Il parametro governa la
# larghezza della banda entro cui l'errore non viene penalizzato, quindi il
# numero di osservazioni che diventano vettori di supporto, quindi il costo.
# Trascriverlo alla lettera renderebbe vettore di supporto quasi ogni riga.
CENTER = {"C": 10.0, "epsilon": 4.0, "gamma": 0.06, "degree": 3}

# `gamma` centrale corrisponde all'impostazione predefinita della libreria su
# questa matrice: con dati standardizzati la varianza media vale uno e il
# valore vale 1 diviso il numero di colonne, cioe' circa 0,056 su 18 variabili.
KERNELS = ("linear", "rbf", "poly")

# Tetto alle iterazioni dell'ottimizzatore. Senza tetto una configurazione mal
# condizionata prosegue fino alla tolleranza per un tempo indeterminato: la
# misura del kernel polinomiale con gamma alto ha richiesto quarantaquattro
# milioni di iterazioni su un terzo delle righe. Il valore e' scelto sopra il
# fabbisogno delle configurazioni che convergono regolarmente, la piu' esigente
# delle quali ne ha richieste tredici milioni a dimensione piena, cosi' che il
# tetto tagli il caso patologico e non quelli legittimi. E' lo stesso
# trattamento gia' applicato ai modelli stimati per discesa coordinata, dove il
# numero massimo di iterazioni e' alzato e le mancate convergenze residue sono
# contate anziche' soppresse.
SVR_MAX_ITER = 20_000_000

# Punti della scala. `None` e' la dimensione piena della parte di addestramento
# della partizione.
LADDER_ROWS = (1000, 2000, 4000, 8000, 12000, None)
QUICK_LADDER_ROWS = (500, 1000, 2000)

# Righe a cui sono misurate la sensibilita' agli iperparametri e la cache.
SENSITIVITY_ROWS = 4000
CACHE_ROWS = 8000
CACHE_SIZES = (200, 500, 1000)

# Assi della sensibilita'. Un asse per volta attorno alla configurazione
# centrale: la misura serve a individuare quale parametro governa il costo, non
# a esplorare la griglia, che non e' ancora fissata.
SENSITIVITY_AXES = {
    "C": [1.0, 10.0, 100.0],
    "epsilon": [1.0, 4.0, 16.0],
    "gamma": [0.01, 0.06, 0.5],
}

# Angolo oneroso plausibile: penalizzazione alta e banda stretta, cioe' molti
# vettori di supporto e ottimizzazione lunga.
CONFIRM = {"C": 100.0, "epsilon": 1.0, "gamma": 0.06, "degree": 3}

# Configurazioni del percettrone. Le architetture sono quelle del laboratorio
# piu' due piu' larghe, perche' la matrice qui ha due ordini di grandezza di
# righe in piu' e la capacita' del laboratorio potrebbe non bastare.
MLP_SPECS = (
    ((32,), 1e-3),
    ((64, 32), 1e-3),
    ((64, 32), 1e-4),
    ((128, 64), 1e-3),
)
MLP_MAX_ITER = 500

# Curva di convergenza del percettrone: architetture e punti di lettura.
MLP_CURVE_SPECS = (
    ((32,), 1e-3),
    ((64, 32), 1e-3),
    ((64, 32), 1e-4),
)
MLP_CURVE_CHECKPOINTS = (100, 250, 500, 1000, 1500, 2000, 3000)

# Angolo in cui la stima del kernel polinomiale si degrada: penalizzazione alta,
# banda stretta, gamma al vertice della griglia candidata.
DEGREE_GAMMAS = (0.15, 0.5)
DEGREES = (2, 3, 4)


def thin(indices: np.ndarray, groups: np.ndarray, target_rows: int | None) -> np.ndarray:
    """Sottocampiona un insieme di righe tenendo un ciclo ogni k dentro ogni motore.

    Il passo e' calcolato sul rapporto fra righe disponibili e righe richieste,
    quindi il numero di righe restituito e' approssimativamente quello chiesto e
    non esattamente quello: i motori hanno durate diverse e il passo e' intero.
    """
    indices = np.sort(np.asarray(indices))
    if target_rows is None or target_rows >= len(indices):
        return indices
    stride = int(np.ceil(len(indices) / target_rows))
    unit_of_row = groups[indices]
    kept = [indices[unit_of_row == unit][::stride] for unit in np.unique(unit_of_row)]
    return np.sort(np.concatenate(kept))


def build_estimator(spec: dict):
    """Costruisce lo stimatore nudo di una misura, dalla sua specifica.

    Lo stimatore e' costruito dentro il processo che lo adatta e non passato
    gia' costruito: la specifica e' un dizionario di tipi elementari, che
    attraversa il confine fra processi senza dipendere da come la libreria
    serializza i propri oggetti.
    """
    kind = spec["kind"]
    if kind == "svr":
        from sklearn.svm import SVR

        return SVR(
            kernel=spec["kernel"],
            C=spec["C"],
            epsilon=spec["epsilon"],
            gamma=spec["gamma"],
            degree=spec["degree"],
            cache_size=spec.get("cache_size", 200),
            max_iter=spec.get("max_iter", SVR_MAX_ITER),
        )
    if kind == "linear_svr":
        from sklearn.svm import LinearSVR

        # Stima in forma primale della sola variante a kernel lineare. Non
        # compare nel laboratorio e va segnalata come tale: risolve lo stesso
        # problema con un ottimizzatore il cui costo cresce linearmente nelle
        # righe, e penalizza anche l'intercetta, quindi non e' lo stesso modello.
        return LinearSVR(
            C=spec["C"],
            epsilon=spec["epsilon"],
            max_iter=spec.get("max_iter", 10_000),
            random_state=MODEL_SEED,
        )
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        # `early_stopping` resta disattivato. La sua partizione interna e'
        # costruita mescolando le righe, quindi collocherebbe cicli adiacenti
        # dello stesso motore da entrambe le parti: e' esattamente la
        # contaminazione che il vincolo di gruppo del protocollo esiste per
        # escludere.
        return MLPRegressor(
            hidden_layer_sizes=tuple(spec["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            learning_rate_init=spec["learning_rate_init"],
            max_iter=spec["max_iter"],
            early_stopping=False,
            random_state=MODEL_SEED,
            warm_start=spec.get("warm_start", False),
            n_iter_no_change=spec.get("n_iter_no_change", 10),
        )
    raise ValueError(f"tipo di stimatore sconosciuto: {kind}")


def _fit_once(spec: dict, X_train, y_train, X_valid, y_valid) -> dict:
    """Adatta una configurazione e ne registra costo, dimensione e errore."""
    from sklearn.exceptions import ConvergenceWarning

    pipeline = build_pipeline(build_estimator(spec))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = pipeline.predict(X_valid)
    predict_seconds = time.perf_counter() - start

    model = pipeline.named_steps["model"]
    support = getattr(model, "support_", None)
    n_iter = getattr(model, "n_iter_", None)
    if isinstance(n_iter, np.ndarray):
        n_iter = float(np.max(n_iter))

    record = {
        "status": "ok",
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_support": float(len(support)) if support is not None else np.nan,
        "support_share": float(len(support) / len(X_train)) if support is not None else np.nan,
        "n_iter": float(n_iter) if n_iter is not None else np.nan,
        "convergence_warnings": sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning)
        ),
    }
    record.update(regression_metrics(y_valid, y_pred))
    return record


def _fit_curve(spec: dict, X_train, y_train, X_valid, y_valid) -> dict:
    """Traiettoria della perdita e dell'errore al crescere delle iterazioni.

    Le iterazioni sono aggiunte a una rete gia' addestrata invece di
    riaddestrarla a ogni punto, quindi l'intera curva costa quanto il solo
    adattamento con il numero massimo di iterazioni e i punti descrivono la
    crescita di una sola rete e non il confronto fra reti diverse.
    """
    from sklearn.exceptions import ConvergenceWarning

    estimator = build_estimator({**spec, "warm_start": True, "max_iter": 0})
    pipeline = build_pipeline(estimator)

    curve = []
    cumulative = 0.0
    previous = 0
    for target in spec["checkpoints"]:
        pipeline.set_params(model__max_iter=target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            cumulative += time.perf_counter() - start

        model = pipeline.named_steps["model"]
        record = {
            "iterations_target": target,
            "n_iter": float(model.n_iter_),
            "loss": float(model.loss_),
            "fit_seconds_cumulative": cumulative,
            "convergence_warnings": sum(
                1 for w in caught if issubclass(w.category, ConvergenceWarning)
            ),
        }
        record.update(regression_metrics(y_valid, pipeline.predict(X_valid)))
        curve.append(record)

        # La rete si e' fermata da sola prima del punto richiesto: i punti
        # successivi ripeterebbero lo stesso stato.
        if model.n_iter_ <= previous:
            break
        previous = model.n_iter_

    return {"status": "ok", "curve": curve}


def _worker(spec, X_train, y_train, X_valid, y_valid, queue) -> None:
    try:
        run = _fit_curve if spec.get("checkpoints") else _fit_once
        queue.put(run(spec, X_train, y_train, X_valid, y_valid))
    except Exception as error:  # la misura fallita e' un esito, non un arresto
        queue.put({"status": f"errore: {type(error).__name__}"})


def measure(spec: dict, data: tuple, timeout: float) -> dict:
    """Esegue una misura in un processo separato, con un limite di tempo.

    Il limite non puo' essere applicato dentro il processo che adatta: la
    stima avviene in codice nativo che non restituisce il controllo
    all'interprete, quindi un segnale resterebbe in attesa fino alla fine
    dell'adattamento, cioe' proprio fino al momento in cui non serve piu'.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(spec, *data, queue))

    start = time.perf_counter()
    process.start()
    result: dict | None = None
    while True:
        try:
            result = queue.get(timeout=0.5)
            break
        except Empty:
            pass
        if time.perf_counter() - start > timeout:
            break
        if not process.is_alive():
            # Il processo e' terminato: l'esito puo' essere ancora in transito
            # sul canale, quindi la lettura viene ritentata una volta prima di
            # dichiarare la misura fallita.
            try:
                result = queue.get(timeout=5.0)
            except Empty:
                pass
            break

    if result is None:
        process.terminate()
        process.join()
        return {"status": "oltre il limite", "fit_seconds": float(timeout)}

    process.join()
    return result


def _row(subset: str, block: str, spec: dict, n_rows: int, result: dict) -> dict:
    """Compone la riga dell'artefatto a partire dalla specifica e dall'esito."""
    row = {
        "subset": subset,
        "block": block,
        "kind": spec["kind"],
        "kernel": spec.get("kernel", ""),
        "C": spec.get("C", np.nan),
        "epsilon": spec.get("epsilon", np.nan),
        "gamma": spec.get("gamma", np.nan),
        "degree": spec.get("degree", np.nan),
        "cache_size": spec.get("cache_size", np.nan),
        "hidden_layer_sizes": str(spec.get("hidden_layer_sizes", "")),
        "learning_rate_init": spec.get("learning_rate_init", np.nan),
        "n_train_rows": n_rows,
    }
    row.update(result)
    return row


def _report(label: str, n_rows: int, result: dict) -> None:
    if result["status"] != "ok":
        print(f"    {label:<44} {n_rows:>6,} righe   {result['status']}")
        return
    support = result.get("support_share", np.nan)
    support_text = "" if np.isnan(support) else f"   vs {support:5.1%}"
    print(
        f"    {label:<44} {n_rows:>6,} righe   {result['fit_seconds']:8.1f} s"
        f"{support_text}   rmse {result['rmse']:.2f}"
    )


def _project(previous: list[tuple[int, float]], n_next: int) -> float:
    """Proietta il costo del punto successivo dai punti gia' misurati.

    L'esponente e' stimato sugli ultimi due punti misurati; con un solo punto
    disponibile si usa 3, che e' l'estremo superiore della crescita nota per
    questo tipo di stimatore ed e' quindi la scelta prudente per una decisione
    che riguarda se tentare o no una misura.
    """
    if not previous:
        return 0.0
    n_last, t_last = previous[-1]
    if len(previous) < 2 or t_last <= 0:
        exponent = 3.0
    else:
        n_prev, t_prev = previous[-2]
        exponent = np.log(t_last / t_prev) / np.log(n_last / n_prev) if t_prev > 0 else 3.0
        exponent = float(np.clip(exponent, 1.0, 3.0))
    return t_last * (n_next / n_last) ** exponent


def run_ladder(subset, split_data, groups, ladder_rows, timeout, budget):
    """Costo di una configurazione centrale al crescere del numero di righe.

    Restituisce le righe dell'artefatto e, per ciascun kernel, il numero di
    righe piu' alto su cui l'adattamento e' terminato entro il limite. E' il
    dato con cui i blocchi successivi evitano di ritentare misure gia'
    dichiarate fuori portata.
    """
    X, y, X_valid, y_valid, train_idx = split_data
    rows = []
    reached: dict[str, int] = {}
    for kernel in KERNELS:
        spec = {"kind": "svr", "kernel": kernel, **CENTER}
        measured: list[tuple[int, float]] = []
        seen: set[int] = set()
        for target in ladder_rows:
            idx = thin(train_idx, groups, target)
            n_rows = len(idx)
            # Due punti richiesti possono ricadere sullo stesso passo di
            # diradamento, e in quel caso la misura sarebbe una ripetizione.
            if n_rows in seen:
                continue
            seen.add(n_rows)
            projected = _project(measured, n_rows)
            if projected > budget:
                rows.append(
                    _row(
                        subset,
                        "ladder",
                        spec,
                        n_rows,
                        {"status": f"non tentato, proiezione {projected:.0f} s"},
                    )
                )
                print(
                    f"    {kernel:<44} {n_rows:>6,} righe   non tentato "
                    f"(proiezione {projected:.0f} s oltre il budget)"
                )
                break
            result = measure(spec, (X.iloc[idx], y[idx], X_valid, y_valid), timeout)
            rows.append(_row(subset, "ladder", spec, n_rows, result))
            _report(f"SVR {kernel}", n_rows, result)
            if result["status"] != "ok":
                break
            measured.append((n_rows, result["fit_seconds"]))
            reached[kernel] = n_rows
    return rows, reached


def _out_of_reach(kernel: str, n_rows: int, reached: dict | None) -> bool:
    """Vero se la scala ha gia' mostrato che quel kernel non arriva a quelle righe."""
    if reached is None:
        return False
    return reached.get(kernel, 0) < n_rows


def run_sensitivity(subset, split_data, groups, timeout, reached=None) -> list[dict]:
    """Effetto di ciascun iperparametro sul costo, un asse per volta."""
    X, y, X_valid, y_valid, train_idx = split_data
    idx = thin(train_idx, groups, SENSITIVITY_ROWS)
    n_rows = len(idx)
    data = (X.iloc[idx], y[idx], X_valid, y_valid)

    rows = []
    for kernel in KERNELS:
        if _out_of_reach(kernel, n_rows, reached):
            spec = {"kind": "svr", "kernel": kernel, **CENTER}
            rows.append(_row(subset, "sensitivity", spec, n_rows, {"status": "non tentato, oltre il limite gia' sulla scala"}))
            print(f"    SVR {kernel:<40} {n_rows:>6,} righe   non tentato (oltre il limite sulla scala)")
            continue
        # La configurazione centrale appartiene a tutti gli assi e verrebbe
        # misurata una volta per asse.
        seen: set[tuple] = set()
        for axis, values in SENSITIVITY_AXES.items():
            if kernel == "linear" and axis == "gamma":
                continue  # il kernel lineare non usa il parametro
            for value in values:
                spec = {"kind": "svr", "kernel": kernel, **CENTER, axis: value}
                signature = tuple(sorted((k, v) for k, v in spec.items() if k != "kind"))
                if signature in seen:
                    continue
                seen.add(signature)
                result = measure(spec, data, timeout)
                rows.append(_row(subset, "sensitivity", spec, n_rows, result))
                _report(f"SVR {kernel}, {axis}={value:g}", n_rows, result)
    return rows


def run_cache(subset, split_data, groups, timeout, reached=None) -> list[dict]:
    """Effetto della dimensione della cache del kernel, a righe fissate."""
    X, y, X_valid, y_valid, train_idx = split_data
    idx = thin(train_idx, groups, CACHE_ROWS)
    n_rows = len(idx)
    data = (X.iloc[idx], y[idx], X_valid, y_valid)

    rows = []
    if _out_of_reach("rbf", n_rows, reached):
        spec = {"kind": "svr", "kernel": "rbf", **CENTER}
        rows.append(_row(subset, "cache", spec, n_rows, {"status": "non tentato, oltre il limite gia' sulla scala"}))
        print(f"    SVR rbf{'':<37} {n_rows:>6,} righe   non tentato (oltre il limite sulla scala)")
        return rows
    for size in CACHE_SIZES:
        spec = {"kind": "svr", "kernel": "rbf", **CENTER, "cache_size": size}
        result = measure(spec, data, timeout)
        rows.append(_row(subset, "cache", spec, n_rows, result))
        _report(f"SVR rbf, cache {size} MB", n_rows, result)
    return rows


def run_confirm(subset, split_data, timeout, reached=None) -> list[dict]:
    """Angolo oneroso a dimensione piena, misurato e non estrapolato.

    Include la stima in forma primale del kernel lineare, che e' l'alternativa
    il cui costo cresce linearmente nelle righe: senza il suo numero accanto
    agli altri la decisione sul trattamento del modello resterebbe fra
    un'alternativa misurata e una supposta.
    """
    X, y, X_valid, y_valid, train_idx = split_data
    data = (X.iloc[train_idx], y[train_idx], X_valid, y_valid)
    n_rows = len(train_idx)

    rows = []
    for kernel in KERNELS:
        spec = {"kind": "svr", "kernel": kernel, **CONFIRM}
        if _out_of_reach(kernel, n_rows, reached):
            # La configurazione centrale non termina a questa dimensione: quella
            # onerosa, che ha penalizzazione dieci volte piu' alta e banda quattro
            # volte piu' stretta, non puo' terminare prima.
            rows.append(_row(subset, "confirm", spec, n_rows, {"status": "non tentato, oltre il limite gia' sulla scala"}))
            print(f"    SVR {kernel:<40} {n_rows:>6,} righe   non tentato (oltre il limite sulla scala)")
            continue
        result = measure(spec, data, timeout)
        rows.append(_row(subset, "confirm", spec, n_rows, result))
        _report(f"SVR {kernel}, angolo oneroso", n_rows, result)

    for C in (1.0, 10.0, 100.0):
        spec = {"kind": "linear_svr", "C": C, "epsilon": CENTER["epsilon"]}
        result = measure(spec, data, timeout)
        rows.append(_row(subset, "confirm", spec, n_rows, result))
        _report(f"LinearSVR, C={C:g}", n_rows, result)
    return rows


def run_mlp(subset, split_data, timeout) -> list[dict]:
    """Costo e convergenza del percettrone a dimensione piena."""
    X, y, X_valid, y_valid, train_idx = split_data
    data = (X.iloc[train_idx], y[train_idx], X_valid, y_valid)
    n_rows = len(train_idx)

    rows = []
    for hidden, lr in MLP_SPECS:
        spec = {
            "kind": "mlp",
            "hidden_layer_sizes": hidden,
            "learning_rate_init": lr,
            "max_iter": MLP_MAX_ITER,
        }
        result = measure(spec, data, timeout)
        rows.append(_row(subset, "mlp", spec, n_rows, result))
        label = f"MLP {hidden}, passo {lr:g}"
        _report(label, n_rows, result)
        if result["status"] == "ok":
            print(
                f"        iterazioni {result['n_iter']:.0f} su {MLP_MAX_ITER}, "
                f"mancate convergenze {result['convergence_warnings']}"
            )
    return rows


def run_degree(subset, split_data, timeout) -> list[dict]:
    """Costo del kernel polinomiale al crescere del grado, sull'angolo peggiore.

    La configurazione tiene fermi penalizzazione alta e banda stretta e fa
    variare grado e ampiezza del kernel: sono i due parametri il cui prodotto
    determina l'ampiezza dei valori della matrice del kernel, e quindi il
    condizionamento del problema che l'ottimizzatore risolve.
    """
    X, y, X_valid, y_valid, train_idx = split_data
    data = (X.iloc[train_idx], y[train_idx], X_valid, y_valid)
    n_rows = len(train_idx)

    rows = []
    for gamma in DEGREE_GAMMAS:
        for degree in DEGREES:
            spec = {
                "kind": "svr",
                "kernel": "poly",
                "C": CONFIRM["C"],
                "epsilon": CONFIRM["epsilon"],
                "gamma": gamma,
                "degree": degree,
            }
            result = measure(spec, data, timeout)
            rows.append(_row(subset, "degree", spec, n_rows, result))
            _report(f"SVR poly, grado {degree}, gamma {gamma:g}", n_rows, result)
            if result["status"] == "ok" and result.get("convergence_warnings"):
                print(f"        tetto alle iterazioni raggiunto: stima troncata")
    return rows


def run_mlp_curve(subset, split_data, timeout) -> list[dict]:
    """Perdita e errore del percettrone al crescere delle iterazioni."""
    X, y, X_valid, y_valid, train_idx = split_data
    data = (X.iloc[train_idx], y[train_idx], X_valid, y_valid)
    n_rows = len(train_idx)

    rows = []
    for hidden, lr in MLP_CURVE_SPECS:
        spec = {
            "kind": "mlp",
            "hidden_layer_sizes": hidden,
            "learning_rate_init": lr,
            "checkpoints": MLP_CURVE_CHECKPOINTS,
            # La rete non deve fermarsi prima del punto richiesto per il solo
            # criterio di miglioramento della perdita: la curva serve a vedere
            # dove quella perdita smette di scendere, non a fermarla.
            "n_iter_no_change": max(MLP_CURVE_CHECKPOINTS),
        }
        result = measure(spec, data, timeout)
        label = f"MLP {hidden}, passo {lr:g}"
        if result["status"] != "ok":
            rows.append(_row(subset, "curve", spec, n_rows, {"status": result["status"]}))
            print(f"    {label:<44} {n_rows:>6,} righe   {result['status']}")
            continue
        for point in result["curve"]:
            rows.append(_row(subset, "curve", spec, n_rows, {"status": "ok", **point}))
            print(
                f"    {label:<32} {point['iterations_target']:>5} iter   "
                f"perdita {point['loss']:9.2f}   rmse {point['rmse']:.2f}   "
                f"{point['fit_seconds_cumulative']:6.1f} s"
            )
    return rows


def merge_artifact(path, frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    """Unisce le righe appena misurate a quelle gia' presenti sul disco.

    Sono sostituite le sole righe dei blocchi rieseguiti; le altre restano.
    Senza questa unione, lanciare un blocco aggiunto dopo cancellerebbe le
    misure precedenti, che su questo esperimento costano ore.
    """
    if not path.exists():
        return frame
    previous = pd.read_csv(path)
    kept = previous[~previous["block"].isin(blocks)]
    return pd.concat([kept, frame], ignore_index=True)


def environment() -> pd.DataFrame:
    """Quantita' da cui si ricava la durata di un esperimento dalle misure.

    Il tempo per adattamento moltiplicato per il numero di configurazioni e per
    il numero di fold, diviso per il numero di processi effettivamente
    utilizzabili, da' la durata di una ricerca. Il numero di processori e la
    memoria vanno percio' registrati accanto ai tempi: gli stessi tempi su una
    macchina diversa portano a una decisione diversa.
    """
    import sklearn

    try:
        total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, AttributeError, OSError):
        total_memory = float("nan")

    return pd.DataFrame(
        [
            {
                "cpu_count": os.cpu_count(),
                "memory_gb": round(total_memory, 1),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            }
        ]
    )


def run_subset(subset, cap, blocks, ladder_rows, timeout, confirm_timeout, budget) -> dict:
    design = build_design(subset, cap=cap)
    split = make_splits(design.groups_train, n_splits=N_SPLITS, seeds=SEARCH_SEEDS)[0]
    groups = np.asarray(design.groups_train)

    split_data = (
        design.X_train,
        np.asarray(design.y_train),
        design.X_train.iloc[split.valid],
        np.asarray(design.y_train)[split.valid],
        np.sort(split.train),
    )

    print(f"\n=== {subset} ===")
    print(
        f"partizione di misura: {len(split.train):,} righe e "
        f"{len(np.unique(groups[split.train]))} motori in addestramento, "
        f"{len(split.valid):,} righe in verifica, {len(design.features)} variabili"
    )

    kernel_rows: list[dict] = []
    mlp_rows: list[dict] = []
    reached: dict | None = None

    if "ladder" in blocks:
        print("\n  costo al crescere delle righe, configurazione centrale")
        ladder_rows_out, reached = run_ladder(
            subset, split_data, groups, ladder_rows, timeout, budget
        )
        kernel_rows += ladder_rows_out
    if "sensitivity" in blocks:
        print(f"\n  sensibilita' agli iperparametri, {SENSITIVITY_ROWS:,} righe richieste")
        kernel_rows += run_sensitivity(subset, split_data, groups, timeout, reached)
    if "cache" in blocks:
        print(f"\n  dimensione della cache del kernel, {CACHE_ROWS:,} righe richieste")
        kernel_rows += run_cache(subset, split_data, groups, timeout, reached)
    if "confirm" in blocks:
        print("\n  angolo oneroso a dimensione piena")
        kernel_rows += run_confirm(subset, split_data, confirm_timeout, reached)
    if "degree" in blocks:
        print("\n  grado del kernel polinomiale sull'angolo peggiore")
        kernel_rows += run_degree(subset, split_data, confirm_timeout)
    if "mlp" in blocks:
        print("\n  percettrone multistrato a dimensione piena")
        mlp_rows += run_mlp(subset, split_data, timeout)
    if "curve" in blocks:
        print("\n  convergenza del percettrone al crescere delle iterazioni")
        mlp_rows += run_mlp_curve(subset, split_data, confirm_timeout)

    outputs = {}
    if kernel_rows:
        outputs["kernel_cost_probe"] = pd.DataFrame.from_records(kernel_rows)
    if mlp_rows:
        outputs["mlp_cost_probe"] = pd.DataFrame.from_records(mlp_rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS_IN_SCOPE))
    parser.add_argument("--cap", type=int, default=RUL_CAP)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["ladder", "sensitivity", "cache", "confirm", "degree", "mlp", "curve"],
        help="blocchi di misura da eseguire",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="limite di tempo di una singola misura, in secondi",
    )
    parser.add_argument(
        "--confirm-timeout",
        type=float,
        default=1800.0,
        help="limite di tempo delle misure a dimensione piena del blocco confirm",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1800.0,
        help="costo proiettato oltre il quale un punto della scala non viene tentato",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="scala ridotta e limiti stretti: convalida della catena",
    )
    args = parser.parse_args()

    ladder_rows = QUICK_LADDER_ROWS if args.quick else LADDER_ROWS
    timeout = 60.0 if args.quick else args.timeout
    confirm_timeout = 60.0 if args.quick else args.confirm_timeout
    budget = 120.0 if args.quick else args.budget

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = environment()
    print(context.to_string(index=False))
    context.to_csv(OUTPUT_DIR / "environment.csv", index=False)

    for subset in args.subsets:
        outputs = run_subset(
            subset, args.cap, args.blocks, ladder_rows, timeout, confirm_timeout, budget
        )
        for name, frame in outputs.items():
            path = OUTPUT_DIR / f"{subset}_{name}.csv"
            merge_artifact(path, frame, args.blocks).to_csv(path, index=False)

    print(f"\nartefatti scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()