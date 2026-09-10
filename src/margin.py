"""Lettura dei modelli a margine e delle reti.

Ruolo nel progetto
    Fornisce al blocco del laboratorio 11 le funzioni con cui se ne commentano i
    modelli, come `src.trees` fa per la famiglia ad albero e `src.nonlinear` per
    il blocco che supera la linearità. Non contiene stimatori né logica di
    valutazione: gli stimatori sono nel registro, la valutazione nel protocollo.

Cosa riceve
    Pipeline già adattate sull'intera parte di addestramento.

Cosa produce
    Riepiloghi strutturali in forma di dizionario, destinati alla tabella delle
    diagnostiche dell'esperimento.

Perché riepiloghi e non coefficienti
    Dei quattro modelli del blocco uno solo ha coefficienti leggibili sulle
    variabili originali, la variante a kernel lineare, la cui funzione stimata
    resta lineare: per quella il registro usa lo stesso lettore dei modelli
    lineari, e la riga della tabella riporta i coefficienti insieme a quelli
    degli altri blocchi. Per le altre tre la funzione stimata non ha
    coefficienti sulle variabili, e ciò che si può leggere è la struttura
    della soluzione.

Vettori di supporto
    Sono le righe che cadono sul bordo della banda di insensibilità o fuori da
    essa, cioè quelle che determinano la soluzione. La loro frazione è la
    misura di complessità propria di questa famiglia e ha la stessa funzione
    che il numero di coefficienti non nulli ha nel blocco lineare e il numero di
    variabili usate in quello ad albero: dice quanta parte dei dati il modello
    sta trattenendo. Una frazione vicina a uno indica che la banda è troppo
    stretta perché il modello riassuma i dati, e che la soluzione è costosa da
    valutare, perché la predizione richiede il calcolo del kernel contro ogni
    vettore di supporto.

    La frazione va letta insieme alla banda selezionata, che la governa
    direttamente: le due quantità sono la stessa lettura vista dal lato del
    parametro e dal lato della soluzione.

Struttura della rete
    Il numero di parametri è la quantità con cui si confronta la capacità di
    architetture diverse, e non coincide con il numero di unità: una rete a due
    strati stretti può avere più parametri di una a strato singolo più largo.
    Il numero di iterazioni eseguite e il valore finale della perdita dicono
    dove la stima si è fermata, che su questo blocco non è una informazione
    accessoria: il numero massimo di iterazioni è un iperparametro in griglia e
    l'arresto per raggiungimento di quel numero è il meccanismo con cui la rete
    viene regolarizzata.

Importanza per permutazione
    Non è definita qui. È la stessa funzione usata dal blocco ad albero, in
    `src.trees`, e riscriverla produrrebbe due definizioni della stessa
    quantità che potrebbero divergere senza che nulla lo segnali. La misura non
    dipende dalla famiglia del modello: riaddestra su ciascuna partizione del
    seme di ricerca, mescola una variabile per volta sulla parte di verifica e
    registra l'aumento dell'errore in cicli.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def support_summary(pipeline: Pipeline) -> dict:
    """Dimensione della soluzione di un modello a margine.

    `support_share` è calcolata sul numero di righe viste in addestramento, che
    è l'intera parte di addestramento del sottoinsieme quando la pipeline
    passata è quella riaddestrata per la lettura.
    """
    model = pipeline.named_steps["model"]
    n_support = int(len(model.support_))
    n_rows = int(model.support_vectors_.shape[0]) if n_support else 0
    n_iter = getattr(model, "n_iter_", None)
    if isinstance(n_iter, np.ndarray):
        n_iter = int(np.max(n_iter))

    return {
        "n_support": n_support,
        "n_support_rows": n_rows,
        "n_iter": int(n_iter) if n_iter is not None else -1,
        "epsilon": float(model.epsilon),
        "intercept": float(np.ravel(model.intercept_)[0]),
    }


def network_summary(pipeline: Pipeline) -> dict:
    """Struttura e punto di arresto di una rete.

    Il numero di parametri conta pesi e intercette di tutti gli strati, cioè i
    gradi di libertà effettivi della funzione stimata.
    """
    model = pipeline.named_steps["model"]
    n_parameters = sum(int(w.size) for w in model.coefs_)
    n_parameters += sum(int(b.size) for b in model.intercepts_)

    return {
        "n_layers": int(model.n_layers_),
        "n_parameters": n_parameters,
        "n_iter": int(model.n_iter_),
        "final_loss": float(model.loss_),
        "stopped_at_max_iter": bool(model.n_iter_ >= model.get_params()["max_iter"]),
    }
