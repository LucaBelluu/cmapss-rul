"""Composizione di pre-processing e modello.

Ruolo nel progetto: costruisce l'oggetto che viene passato al protocollo di
valutazione. Tutto cio' che precede il modello (selezione delle colonne,
standardizzazione) sta dentro la pipeline e viene quindi adattato sulla sola
parte di addestramento di ogni fold, mai sull'intero insieme.

Riceve: uno stimatore di regressione e, opzionalmente, l'elenco delle colonne
da usare. Produce: una Pipeline di scikit-learn.

La standardizzazione e' applicata a tutti i modelli, anche a quelli per cui e'
irrilevante (alberi e insiemi di alberi). Una pipeline differenziata per
famiglia introdurrebbe una differenza di condizioni fra modelli confrontati,
che e' esattamente cio' che il protocollo deve escludere; il costo sugli
alberi e' trascurabile.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(estimator, *, columns: list[str] | None = None, scale: bool = True) -> Pipeline:
    """Compone selezione delle colonne, standardizzazione e modello.

    columns limita la matrice alle colonne indicate ed e' usato dalle baseline,
    che devono vedere un sottoinsieme delle variabili disponibili senza che la
    matrice di progetto venga ricostruita in modo diverso.
    """
    steps = []
    if columns is not None:
        steps.append(
            ("select", ColumnTransformer([("keep", "passthrough", columns)], remainder="drop"))
        )
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)