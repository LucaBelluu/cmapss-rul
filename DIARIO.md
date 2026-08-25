# Diario di progetto — cmapss-rul

Progetto sperimentale di Machine Learning, A.A. 2025/2026.
Dataset: NASA C-MAPSS (Turbofan Engine Degradation Simulation), dal NASA
Prognostics Data Repository.
Task: regressione della vita utile residua (Remaining Useful Life) di
motori aeronautici a partire da letture di sensori.

## Obiettivo e domande di analisi

La consegna richiede di analizzare, studiare e addestrare modelli di
machine learning sul dataset scelto, e di individuare con le tecniche di
model selection e validazione studiate nel corso il modello migliore nel
predire la variabile target. Il deliverable è una repository pubblica.

Le domande che il lavoro intende chiudere:

1. **Confronto tra famiglie di modelli.** Sotto un protocollo di
   valutazione unico e a parità di dati e di condizioni, quale famiglia
   di modelli (lineare regolarizzata, additiva non lineare, basata su
   alberi, kernel, reti neurali) predice meglio la vita utile residua, e
   di quanto il vantaggio supera la variabilità tra le partizioni?

2. **Costo della complessità.** Il guadagno predittivo dei modelli più
   complessi giustifica la perdita di interpretabilità rispetto a una
   baseline lineare regolarizzata?

3. **Effetto della struttura del dato.** Quanto incide sulla stima delle
   prestazioni il fatto che le osservazioni siano cicli di funzionamento
   raggruppati per motore, anziché osservazioni indipendenti?

---

## [25-08-2026] — Impostazione del progetto e perimetro delle tecniche

### Scelta del dataset e del task

Ho scelto NASA C-MAPSS, dal NASA Prognostics Data Repository, e il task
di regressione.

Motivo: è un dataset pubblico, documentato e ampiamente studiato in
letteratura, con una struttura non banale. Ogni osservazione è un ciclo
di funzionamento di un motore seguito dall'inizio del monitoraggio fino
al guasto, con 3 impostazioni operative e 21 letture di sensori. Questo
permette di applicare per intero il repertorio di regressione del corso e
di affrontare un problema di validazione reale, non artificiale.

La variabile target è la Remaining Useful Life, cioè il numero di cicli
di funzionamento che mancano al guasto. Non è una colonna presente nei
file: va costruita.

### Perimetro delle tecniche

Ho fissato il perimetro del confronto sul programma del corso:

- regressione lineare semplice e multipla;
- metodi di ricampionamento: validation set approach, LOOCV, K-Fold,
  bootstrap;
- selezione del modello e regolarizzazione: best subset selection,
  forward e backward stepwise, Ridge, Lasso, Elastic Net, PCA e
  Principal Component Regression;
- superamento della linearità: regressione polinomiale, step functions,
  spline, GAM;
- alberi di regressione con pruning per cost-complexity, bagging;
- ensemble: random forest, AdaBoost, gradient boosting, XGBoost;
- Support Vector Regression con kernel lineare, radiale e polinomiale;
- reti neurali multistrato per la regressione.

Escluso KNN in versione regressiva. Motivo: nel corso KNN è trattato
esclusivamente come classificatore, e il ruolo di modello non parametrico
nel confronto è già coperto da alberi ed ensemble. Includerlo avrebbe
portato in tabella un modello non presentato per questo task, senza
guadagno informativo.

I metodi non supervisionati (PCA, K-Means, clustering gerarchico)
rientrano nel lavoro come strumenti di esplorazione e di riduzione della
dimensionalità, non come modelli in confronto: il task è di regressione e
il confronto riguarda la capacità predittiva sul target.

### Struttura del dato e conseguenze sulla validazione

I dataset usati nei laboratori del corso impiegano partizioni casuali per
riga su osservazioni indipendenti. C-MAPSS non ha questa proprietà: ogni
riga è un ciclo di funzionamento di un motore specifico, e le righe dello
stesso motore sono fortemente correlate tra loro perché descrivono la
stessa traiettoria di degrado a cicli consecutivi.

Una partizione casuale per riga collocherebbe cicli adiacenti dello
stesso motore in partizioni diverse, permettendo al modello di essere
valutato su osservazioni quasi identiche a quelle su cui è stato
addestrato. Il risultato sarebbe una stima delle prestazioni
sistematicamente ottimistica e priva di significato predittivo.

La partizione dovrà quindi avvenire per unità motore, mantenendo tutti i
cicli di uno stesso motore nella stessa partizione. Questo comporta l'uso
di schemi di cross-validation con vincolo di gruppo, che non compaiono
nei laboratori del corso ma sono la trasposizione diretta del K-Fold a
dati raggruppati.

Stato: impostazione e perimetro definiti. Nessun codice ancora scritto.

---

## [25-08-2026] — Ambiente di sviluppo

### Gestione dell'ambiente

Ho creato la cartella di progetto in `/Users/lucabellu/cmapss-rul` e un
ambiente conda dedicato di nome `cmapss-rul` con Python 3.12.14, tramite
la distribuzione Miniforge già presente sulla macchina e configurata con
`conda-forge` come canale predefinito.

Motivo della scelta di conda anziché di un ambiente virtuale `venv`:
entrambi isolano le librerie allo stesso modo, ma conda gestisce anche
l'interprete Python, che viene installato dentro l'ambiente. Con `venv`
sarebbe stato necessario installare separatamente un Python recente a
livello di sistema, perché quello fornito da macOS è troppo vecchio per
lo stack usato.

Ho fissato Python 3.12 e non una versione più recente perché è quella su
cui ho verificato il funzionamento congiunto dell'intero stack.

### Installazione delle librerie

Ho installato le librerie con `pip` dentro l'ambiente conda, riservando a
conda la gestione dell'interprete e delle dipendenze native non Python.

Alternative scartate:

- **Tutte le librerie da conda-forge, con `environment.yml` come
  manifesto.** Scartata perché `conda env export` produce un file con le
  build string legate all'architettura della macchina, non installabile
  altrove, mentre la variante `--from-history` perde i pin delle
  dipendenze transitive. Il deliverable è una repository pubblica e il
  manifesto dell'ambiente deve essere leggibile e installabile da
  chiunque.
- **Ripartizione sistematica tra conda-forge per lo stack scientifico e
  pip per il resto.** Scartata perché mescolare i due gestori sullo
  stesso insieme di pacchetti è una causa nota di ambienti che si
  corrompono agli aggiornamenti, dato che conda non ha visibilità su
  quanto installato da pip, e perché richiederebbe due manifesti anziché
  uno.

Librerie installate: numpy, pandas, scipy, scikit-learn, statsmodels,
xgboost, pygam, matplotlib, seaborn, jupyterlab, ipykernel.

### Runtime OpenMP per XGBoost

- **Sintomo.** L'importazione di `xgboost` fallisce con
  `XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded`,
  causato da `Library not loaded: @rpath/libomp.dylib`. Tutte le altre
  librerie dello stack risultano importabili.
- **Causa.** XGBoost non è puro Python: la sua parte computazionale è una
  libreria nativa che richiede il runtime OpenMP per la
  parallelizzazione. La distribuzione su PyPI include la libreria nativa
  ma non OpenMP, che è una dipendenza di sistema fuori dalla portata di
  pip. Su macOS il runtime non è presente per impostazione predefinita.
- **Soluzione.** Installazione di `llvm-openmp` da conda-forge
  nell'ambiente del progetto, che colloca `libomp.dylib` in
  `$CONDA_PREFIX/lib/`, uno dei percorsi in cui XGBoost cerca la libreria
  al caricamento.
- **Alternativa scartata.** L'installazione di OpenMP tramite Homebrew,
  che è la via indicata dal messaggio d'errore e dalla documentazione di
  XGBoost. Scartata perché colloca una dipendenza del progetto a livello
  di sistema anziché dentro l'ambiente, rendendo la riproduzione
  dipendente da uno stato della macchina che i file della repository non
  descrivono.
- **Residuo.** Il manifesto delle librerie Python non descrive da solo
  l'ambiente completo: la riproduzione richiede anche l'installazione di
  `llvm-openmp` da conda. Il passaggio va documentato nella sezione di
  riproduzione del README.

Questo episodio ha corretto un presupposto errato dell'impostazione
iniziale, secondo cui lo stack non conteneva componenti native
problematiche su macOS.

### Verifica funzionale dello stack

Ho verificato che le librerie funzionino insieme, non soltanto che siano
installate. La verifica esercita i punti a rischio: le API di
scikit-learn introdotte nelle versioni recenti, lo schema di
cross-validation con vincolo di gruppo su cui si reggerà il protocollo di
valutazione, l'addestramento di un GAM, e l'addestramento di un
regressore XGBoost.

Versioni verificate:

| Libreria | Versione |
|---|---|
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| xgboost | 3.4.1 |
| pygam | 0.12.0 |
| statsmodels | 0.14.6 |

Esiti: `SplineTransformer` e `root_mean_squared_error` disponibili e
funzionanti, `GroupKFold` operativo, `LinearGAM` addestrato senza errori
sotto numpy 2.x, `XGBRegressor` addestrato e in grado di produrre
predizioni.

Il controllo su `pygam` era il più rilevante, essendo la dipendenza dello
stack con i vincoli di versione più stretti e il minor livello di
manutenzione.

Stato: ambiente creato e verificato.

### Manifesto delle dipendenze

Ho fissato le versioni delle librerie in `requirements.txt`, generato a
partire dall'ambiente effettivo anziché scritto a mano, così da includere
anche le dipendenze transitive.

Il file prodotto da `pip freeze` conteneva una riga inutilizzabile nella
forma `packaging @ file:///percorso/di/build`. La causa è che il pacchetto
`packaging` proviene dalla distribuzione conda del Python dell'ambiente e
non da PyPI: pip non dispone di un indice da cui recuperarlo e ripiega sul
percorso locale da cui è stato costruito, che non esiste su nessun'altra
macchina. Una singola riga di questo tipo interrompe l'installazione
dell'intero file.

Ho generato il manifesto con `pip list --format=freeze`, che produce
sempre pin nella forma `nome==versione`, ed escluso `pip`, `setuptools` e
`wheel`, che sono infrastruttura dell'ambiente e non dipendenze del
progetto.

Ho registrato l'ambiente come kernel Jupyter con nome `cmapss-rul`, per
evitare che i notebook vengano eseguiti con un interprete diverso da
quello del progetto.

### Struttura della repository

Ho organizzato la repository separando il codice riutilizzabile (`src/`),
l'orchestrazione degli esperimenti (`scripts/`), l'analisi e la
narrazione (`notebooks/`), i dati (`data/`, con `raw/` per i file
originali e `interim/` per i derivati), gli artefatti generati
(`experiments/`) e gli output finali (`results/figures/` e
`results/tables/`).

Motivo: separare come si esegue una singola operazione da quali
operazioni compongono un esperimento rende un esperimento riproducibile
leggendo i comandi che lo compongono. Separare l'esecuzione dalla
narrazione permette di eseguire i notebook dall'inizio alla fine in pochi
secondi, perché leggono artefatti già prodotti e non ricalcolano nulla.

Le cartelle ancora vuote contengono un file `.gitkeep`, perché git
versiona file e non cartelle: senza segnaposto la struttura non
comparirebbe nella repository.

### Regole di esclusione dal versionamento

Restano fuori dal versionamento i dati grezzi, gli artefatti degli
esperimenti, i modelli serializzati, gli archivi compressi, la cache
Python, i checkpoint dei notebook e i file di sistema di macOS. Entrano
nel versionamento le tabelle e le figure finali in `results/`, che sono
leggere e costituiscono la prova tracciabile dei risultati consultabile
senza eseguire il codice.

Per `data/` ed `experiments/` ho usato la forma `data/*` seguita da
`!data/.gitkeep` anziché la più breve `data/`. Motivo: con `data/` git
ignora l'intera cartella e non ne esamina il contenuto, quindi nessuna
eccezione al suo interno può essere applicata; con `data/*` git continua
a valutare i singoli percorsi e l'eccezione sul segnaposto funziona.

Nota tecnica sulla verifica: `git check-ignore -v` riporta l'ultima
regola che combacia anche quando questa è una negazione, e restituisce
codice di uscita zero in entrambi i casi. Non è quindi un test
affidabile per distinguere un file escluso da uno esplicitamente
reincluso. La verifica affidabile consiste nell'ispezionare cosa entra
davvero nell'area di stage.

### Acquisizione del dataset

Ho scaricato l'archivio del dataset dal NASA Prognostics Data Repository
e collocato i file estratti in `data/raw/`, senza cartelle intermedie e
senza modificarli. I file grezzi restano immutati: ogni trasformazione
produrrà file separati in `data/interim/`, così che l'origine resti
sempre distinguibile dal derivato. Insieme ai dati ho conservato la
documentazione originale del dataset, cioè `readme.txt` e il documento
sulla modellazione della propagazione del danno.

L'acquisizione è manuale e non automatizzata. Limite dichiarato: la
repository non contiene una procedura eseguibile per ottenere i dati, e
chi la clona deve seguire le istruzioni di acquisizione documentate nel
README, che riportano l'indirizzo della sorgente e la struttura attesa
dei file. Ho valutato e poi scartato uno script di acquisizione: sarebbe
rimasto nella repository senza essere mai stato eseguito, e uno script
non testato dà l'apparenza di una procedura riproducibile senza esserlo.

Verifica dei file collocati in `data/raw/`:

| File | Righe | Unità | Colonne |
|---|---|---|---|
| train_FD001.txt | 20631 | 100 | 26 |
| train_FD002.txt | 53759 | 260 | 26 |
| train_FD003.txt | 24720 | 100 | 26 |
| train_FD004.txt | 61249 | 249 | 26 |
| test_FD001.txt | 13096 | 100 | 26 |
| test_FD002.txt | 33991 | 259 | 26 |
| test_FD003.txt | 16596 | 100 | 26 |
| test_FD004.txt | 41214 | 248 | 26 |

| File | Righe |
|---|---|
| RUL_FD001.txt | 100 |
| RUL_FD002.txt | 259 |
| RUL_FD003.txt | 100 |
| RUL_FD004.txt | 248 |

Le 26 colonne corrispondono a identificativo dell'unità, numero di ciclo,
3 impostazioni operative e 21 letture di sensori. Il numero di righe di
ciascun file di RUL coincide con il numero di unità del corrispondente
file di test, come atteso: ogni motore di test è troncato prima del
guasto e ha una sola etichetta di riferimento, che indica la vita utile
residua all'ultimo ciclo osservato. Nei file di training le traiettorie
arrivano invece al guasto, quindi la vita utile residua a ogni ciclo si
ricava per differenza dall'ultimo ciclo della stessa unità.

I quattro sottoinsiemi hanno dimensioni molto diverse: 709 motori di
training in totale, di cui 100 in FD001 e in FD003, 260 in FD002 e 249 in
FD004, per circa 160.000 cicli complessivi. Il numero di motori, e non il
numero di righe, è la dimensione campionaria rilevante, perché il motore
è l'unità di partizionamento imposta dalla struttura del dato. Cento
unità sono una base ristretta per una cross-validation stabile, e questo
vincola la scelta dei sottoinsiemi da utilizzare.

Ho verificato le regole di esclusione contro i file reali dopo la loro
collocazione: nessun file di dati compare tra quelli tracciati.