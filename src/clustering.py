"""Raggruppamento delle traiettorie per motore.

Ruolo nel progetto
    Strumento di commento e non una riga del confronto. Il task del progetto e'
    di regressione e i metodi non supervisionati del laboratorio 12 entrano come
    strumenti di esplorazione: servono a leggere l'asimmetria fra i due
    sottoinsiemi in perimetro, che hanno rispettivamente uno e due modi di
    guasto, e a dire se quella differenza sia visibile nella forma delle
    traiettorie oltre che nei punteggi dei modelli.

Cosa riceve
    Una struttura `Design`, cioe' la stessa matrice su cui il confronto e' stato
    condotto. Il perimetro delle variabili e' quindi identico a quello dei
    modelli e non viene ridefinito qui.

Cosa produce
    Una tabella di variabili per motore, i punteggi dei raggruppamenti al variare
    del numero di gruppi e del metodo, le etichette assegnate a ciascun motore e
    la matrice di aggregazione da cui si disegna il dendrogramma. Non scrive su
    disco.

Unita' di osservazione
    Il motore, non il ciclo. Il confronto fra modelli lavora su una riga per
    ciclo perche' predice la vita residua a ogni ciclo; qui la domanda riguarda
    la forma della traiettoria nel suo complesso, quindi ogni motore e' un punto.

    Sono usate le sole traiettorie di addestramento. Quelle di verifica sono
    troncate in un punto casuale prima del guasto, quindi le statistiche di fine
    vita non vi sono definite e la durata osservata non e' la durata del motore.

Variabili per motore
    Per ciascun sensore non costante, la lettura media negli ultimi cicli e la
    deriva totale, cioe' la differenza fra la media degli ultimi cicli e quella
    dei primi. La prima descrive lo stato al guasto, la seconda quanto il sensore
    si e' spostato lungo la vita del motore. A queste si aggiunge la durata della
    traiettoria.

    Le finestre iniziale e finale non si sovrappongono su nessun motore: la
    traiettoria piu' breve dei due sottoinsiemi in perimetro dura 128 cicli.

    Le impostazioni operative sono escluse. Su FD001 e FD003 il regime di volo e'
    unico e la loro variazione residua e' oscillazione di misura attorno a un
    valore fisso: la standardizzazione, che il calcolo delle distanze richiede,
    la porterebbe a scala piena e ne farebbe rumore dentro la distanza, su un
    insieme di cento punti. L'esclusione non e' una disparita' di trattamento
    rispetto al confronto, dove le due variabili sono mantenute, perche' il
    raggruppamento non e' una riga del confronto.

    Il numero di ciclo non entra come variabile: la sua informazione a livello di
    motore e' la durata della traiettoria, che e' gia' presente.

Standardizzazione
    Applicata a tutte le variabili prima di ogni calcolo, come nel laboratorio.
    Le letture dei sensori hanno scale che differiscono di ordini di grandezza e
    senza standardizzazione la distanza euclidea sarebbe determinata dalle sole
    variabili di ampiezza maggiore.

Etichette esterne
    Il modo di guasto di ciascun motore non e' distribuito con il dataset, quindi
    non esiste un'etichetta di riferimento contro cui misurare i raggruppamenti
    dentro un sottoinsieme. L'indice di Rand corretto e' percio' usato dentro il
    sottoinsieme come misura di accordo fra due raggruppamenti diversi, e non
    come misura di correttezza. Sulla versione che unisce i due sottoinsiemi
    esiste invece un'etichetta esterna vera, l'appartenenza al sottoinsieme, e
    li' l'indice ha il significato che ha nel laboratorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import CYCLE_COL, SETTING_COLS

# Ampiezza delle finestre di inizio e fine traiettoria, in cicli. Non si
# sovrappongono su nessun motore: la traiettoria piu' breve dei sottoinsiemi in
# perimetro dura 128 cicli.
WINDOW = 10

# Numeri di gruppi esplorati. Parte da 2 perche' con un gruppo solo la silhouette
# non e' definita, e arriva a 6 perche' oltre, su cento unita', i gruppi
# scendono sotto la ventina di elementi e la misura diventa instabile.
N_CLUSTERS = (2, 3, 4, 5, 6)

# Criteri di aggregazione del clustering gerarchico, gli stessi del laboratorio.
LINKAGES = ("ward", "complete", "average")

# Seme del raggruppamento. E' distinto dai semi del protocollo, che governano il
# partizionamento: questo riguarda l'inizializzazione dei centroidi.
CLUSTER_SEED = 0
CLUSTER_SEEDS = (0, 1, 2, 3, 4)

# Numero di inizializzazioni di K-Means, come nel laboratorio.
N_INIT = 10


def engine_features(design) -> pd.DataFrame:
    """Variabili per motore, calcolate sulle traiettorie di addestramento.

    Ritorna un DataFrame indicizzato per identificativo di unita', con la durata
    della traiettoria e, per ciascun sensore, la lettura media di fine vita e la
    deriva totale.
    """
    frame = design.X_train.copy()
    frame.insert(0, "unit", design.groups_train)

    sensors = [
        c for c in design.features if c != CYCLE_COL and c not in SETTING_COLS
    ]

    records = []
    for unit, trajectory in frame.groupby("unit", sort=True):
        trajectory = trajectory.sort_values(CYCLE_COL)
        head = trajectory.head(WINDOW)[sensors].mean()
        tail = trajectory.tail(WINDOW)[sensors].mean()

        record = {"unit": int(unit), "durata": int(trajectory[CYCLE_COL].max())}
        for sensor in sensors:
            record[f"{sensor}_finale"] = float(tail[sensor])
            record[f"{sensor}_deriva"] = float(tail[sensor] - head[sensor])
        records.append(record)

    return pd.DataFrame.from_records(records).set_index("unit").sort_index()


def standardize(features: pd.DataFrame) -> np.ndarray:
    """Matrice standardizzata su cui operano proiezione e raggruppamenti."""
    return StandardScaler().fit_transform(features.to_numpy(dtype=float))


def projection(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Proiezione sulle prime componenti principali, per la visualizzazione.

    La proiezione non entra nel calcolo dei raggruppamenti, che operano sulla
    matrice intera: serve a rappresentare in due dimensioni un risultato ottenuto
    in trentuno, e la quota di varianza trattenuta dice quanto quella
    rappresentazione sia fedele.
    """
    pca = PCA(n_components=n_components, random_state=CLUSTER_SEED)
    coordinates = pca.fit_transform(matrix)
    return coordinates, pca.explained_variance_ratio_


def cluster_scores(matrix: np.ndarray, labels_reference: np.ndarray | None = None) -> pd.DataFrame:
    """Punteggi di K-Means e del gerarchico al variare del numero di gruppi.

    silhouette
        Misura interna: usa le sole distanze e le etichette prodotte.
    ari_vs_kmeans
        Accordo fra il raggruppamento gerarchico e quello di K-Means allo stesso
        numero di gruppi. Non e' una misura di correttezza, perche' il modo di
        guasto dei motori non e' distribuito con il dataset e non esiste
        un'etichetta di riferimento.
    ari_vs_reference
        Presente solo quando esiste un'etichetta esterna vera, cioe' sulla
        versione che unisce i due sottoinsiemi.
    """
    records = []
    for k in N_CLUSTERS:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=CLUSTER_SEED)
        km_labels = km.fit_predict(matrix)
        rows = [("kmeans", "", km_labels)]
        for link in LINKAGES:
            model = AgglomerativeClustering(n_clusters=k, linkage=link)
            rows.append(("agglomerative", link, model.fit_predict(matrix)))

        for method, link, labels in rows:
            record = {
                "n_clusters": k,
                "metodo": method,
                "linkage": link,
                "silhouette": float(silhouette_score(matrix, labels)),
                "ari_vs_kmeans": float(adjusted_rand_score(km_labels, labels)),
                "dimensione_minima": int(np.bincount(labels).min()),
                "dimensione_massima": int(np.bincount(labels).max()),
            }
            if labels_reference is not None:
                record["ari_vs_reference"] = float(
                    adjusted_rand_score(labels_reference, labels)
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def cluster_labels(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Etichette assegnate a ciascuna unita', per ogni numero di gruppi esplorato.

    Sono prodotte per K-Means e per il gerarchico con aggregazione di Ward, che
    e' quella che il laboratorio confronta con K-Means. Le etichette di un
    raggruppamento non hanno ordine ne' significato: solo la partizione che
    inducono e' interpretabile.
    """
    table = pd.DataFrame(index=index)
    for k in N_CLUSTERS:
        table[f"kmeans_k{k}"] = KMeans(
            n_clusters=k, n_init=N_INIT, random_state=CLUSTER_SEED
        ).fit_predict(matrix)
        table[f"ward_k{k}"] = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(matrix)
    return table


def seed_stability(matrix: np.ndarray) -> pd.DataFrame:
    """Sensibilita' di K-Means al seme di inizializzazione dei centroidi.

    L'algoritmo converge a un minimo locale e semi diversi possono produrre
    partizioni diverse. La colonna dell'accordo confronta ciascuna partizione con
    quella del seme di riferimento: un valore prossimo a uno dice che la
    soluzione trovata e' stabile e non un artefatto dell'inizializzazione.
    """
    records = []
    for k in N_CLUSTERS:
        reference = None
        for seed in CLUSTER_SEEDS:
            labels = KMeans(
                n_clusters=k, n_init=N_INIT, random_state=seed
            ).fit_predict(matrix)
            if reference is None:
                reference = labels
            records.append(
                {
                    "n_clusters": k,
                    "seme": seed,
                    "silhouette": float(silhouette_score(matrix, labels)),
                    "ari_vs_seme_riferimento": float(adjusted_rand_score(reference, labels)),
                }
            )
    return pd.DataFrame.from_records(records)


def linkage_matrix(matrix: np.ndarray, method: str = "ward") -> pd.DataFrame:
    """Matrice di aggregazione del clustering gerarchico, da cui si disegna il dendrogramma.

    Viene calcolata qui e non nel notebook perche' il notebook legge artefatti e
    non esegue lavoro: la regola vale anche quando il costo e' trascurabile,
    altrimenti la figura dipenderebbe da un calcolo non registrato.
    """
    Z = linkage(matrix, method=method, metric="euclidean")
    return pd.DataFrame(Z, columns=["nodo_1", "nodo_2", "distanza", "dimensione"])