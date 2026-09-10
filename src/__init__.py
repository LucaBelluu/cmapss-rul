"""Codice riutilizzabile del progetto cmapss-rul.

I moduli di questo pacchetto contengono le operazioni elementari (caricamento
dei dati, costruzione delle variabili, addestramento, valutazione) e sono
parametrici: non decidono da soli quali esperimenti eseguire. La composizione
degli esperimenti sta in `scripts/`, l'analisi dei risultati in `notebooks/`.

Il pacchetto è pensato per essere importato con la radice della repository come
directory di lavoro, ad esempio con `python -m scripts.nome_script`.
"""