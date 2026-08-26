# Diario di progetto — cmapss-rul

Progetto sperimentale di Machine Learning, A.A. 2025/2026.
Dataset: NASA C-MAPSS (Turbofan Engine Degradation Simulation), dal NASA
Prognostics Data Repository.
Task: regressione della vita utile residua (Remaining Useful Life) di
motori aeronautici a partire da letture di sensori.

## Obiettivo

La consegna richiede di analizzare, studiare e addestrare modelli di
machine learning sul dataset scelto. Tutti i modelli per la regressione
visti durante il corso devono essere confrontati e commentati. Il
deliverable è una repository pubblica.

Il lavoro è quindi vincolato a due esiti congiunti: un confronto che
copra per intero il repertorio di regressione del programma, condotto
sotto un protocollo di valutazione unico e a parità di dati e di
condizioni; e un commento di ciascun modello che ne legga il
comportamento su questo dataset, non limitato al valore della metrica.

Il dataset impone un vincolo che il confronto deve rispettare: le
osservazioni sono cicli di funzionamento raggruppati per motore e non
sono indipendenti. Il protocollo di valutazione è costruito di
conseguenza.

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

## [25-08-2026] — Messa sotto controllo di versione e pubblicazione della repository

Ho configurato l'identità di autore a livello locale della repository e
non globale, così da non alterare la configurazione degli altri progetti
presenti sulla macchina. La configurazione precede il primo commit
perché l'autore viene inciso al momento della creazione del commit e
modificarlo dopo richiede di riscrivere la cronologia.

Ho registrato il primo commit con i dieci file dell'impostazione
iniziale: regole di esclusione, diario, manifesto delle dipendenze e
segnaposto delle cartelle. Verificato che nessun file di dati sia entrato
nella cronologia.

Ho creato la repository remota pubblica senza inizializzarla con file
predefiniti, perché un commit iniziale generato dal servizio remoto
avrebbe una cronologia disgiunta da quella locale e impedirebbe il primo
invio. Autenticazione via token personale con ambito limitato alla
gestione delle repository, generato specificamente per questo progetto
anziché riutilizzarne uno esistente.

Repository allineata al remoto. Verificato dall'interfaccia web che la
cartella dei dati contenga il solo segnaposto.

Imperfezione registrata: due commit consecutivi sul diario portano lo
stesso messaggio pur contenendo modifiche diverse. Ho scelto di non
riscrivere la cronologia per correggerlo, dato che l'operazione avrebbe
un costo superiore al difetto.

La scrittura del README è rimandata alla fase conclusiva del lavoro, per
poterlo redigere sul progetto completo anziché su ipotesi. Limite
dichiarato: fino ad allora la repository è pubblica ma priva di una
descrizione leggibile dall'esterno.

## [25-08-2026] — CORREZIONE: formulazione della consegna e obiettivo del lavoro

CORREZIONE: l'intestazione del diario riportava la consegna in una forma
non corrispondente al testo ufficiale e vi aggiungeva tre domande di
analisi formulate in proprio.

Il testo ufficiale della consegna è: analizzare, studiare e addestrare
modelli di machine learning sul dataset scelto; tutti i modelli per la
regressione visti durante il corso devono essere confrontati e
commentati.

Cosa cambia rispetto alla formulazione precedente:

- il perimetro dei modelli non è una scelta di progetto ma un obbligo di
  copertura: l'assenza dal confronto di un modello di regressione
  presente nel programma è una consegna incompleta;
- il commento di ciascun modello ha lo stesso peso del confronto
  numerico, e non è un complemento della tabella dei risultati;
- la consegna non chiede di formulare domande di ricerca proprie. Le tre
  domande che comparivano in intestazione sono state rimosse: erano un
  livello interpretativo aggiunto sopra la consegna, e mantenerle avrebbe
  spostato il lavoro verso la risposta a quelle domande anziché verso la
  copertura richiesta.

Ho conservato come criteri di conduzione, e non come domande, il
confronto a parità di condizioni e la lettura del divario tra modelli in
rapporto alla variabilità della stima: sono il modo in cui il confronto
richiesto viene reso difendibile, non un obiettivo aggiuntivo.

Motivo della correzione per voce nuova anziché per modifica della voce
originale: le voci cronologiche già scritte non si riscrivono.
L'intestazione, che non è cronologica, è stata invece sostituita.

## [25-08-2026] — CORREZIONE: perimetro delle tecniche fissato sui laboratori del corso

CORREZIONE: il perimetro registrato in precedenza era una ricostruzione a
memoria del programma. Poiché la consegna impone il confronto di tutti i
modelli di regressione visti nel corso, l'elenco è stato riscritto sui
notebook dei laboratori, che sono la fonte di ciò che il corso ha
trattato. I laboratori di classificazione non entrano nel perimetro,
perché il task del progetto è di regressione.

### Modelli da confrontare

| Modello | Laboratorio |
|---|---|
| Regressione lineare semplice e multipla (OLS) | 3 |
| Ridge | 7 |
| Lasso | 7 |
| Elastic Net | 7 |
| Principal Components Regression | 7 |
| Regressione polinomiale | 8 |
| Step functions | 8 |
| Regression spline su base B-spline | 8 |
| Generalized Additive Model | 8 |
| Albero di regressione con pruning per cost-complexity | 9 |
| Bagging di alberi | 9 |
| Random Forest | 10 |
| AdaBoost | 10 |
| Gradient Boosting | 10 |
| XGBoost | 10 |
| Support Vector Regression, kernel lineare | 11 |
| Support Vector Regression, kernel radiale | 11 |
| Support Vector Regression, kernel polinomiale | 11 |
| Rete neurale multistrato (MLP) | 11 |

Le spline del corso sono regression spline su base B-spline, con numero
di nodi e grado come iperparametri: non smoothing spline né natural
spline. Le step functions non corrispondono a una classe di libreria e
vanno costruite come variabili indicatrici su intervalli della variabile.

### Selezione delle variabili

Best subset selection, forward stepwise selection, backward stepwise
selection (laboratorio 7). La backward stepwise nel materiale è proposta
come esercizio e non svolta: va implementata interamente.

### Ricampionamento e validazione

Validation set approach, LOOCV, K-Fold, bootstrap (laboratorio 6). Il
bootstrap del corso è una funzione di ricampionamento scritta da zero,
usata per stimare la variabilità di una statistica e non come procedura
di addestramento. Qui il ricampionamento avviene sui motori e non sulle
righe, per la stessa ragione per cui il partizionamento avviene per
unità.

### Conduzione degli esperimenti

Composizione di pre-processing e modello in una pipeline unica,
standardizzazione delle variabili, selezione degli iperparametri per
ricerca esaustiva su griglia valutata in cross-validation (laboratori 7 e
11). Collocano il pre-processing dentro il flusso di validazione anziché
prima di esso.

Metriche di regressione del corso: MSE, RMSE, MAE, R quadro.

### Lettura dei modelli

Percorso dei coefficienti al variare della penalizzazione (laboratorio
7), funzioni parziali dei termini del GAM (laboratorio 8), importanza
delle variabili da riduzione di impurità e per permutazione (laboratorio
10). Sono gli strumenti con cui viene prodotto il commento di ciascun
modello richiesto dalla consegna.

### Metodi non supervisionati

PCA, K-Means, clustering gerarchico (laboratorio 12): strumenti di
esplorazione e di riduzione della dimensionalità, non modelli in
confronto.

### Esclusione di KNN regressivo

Confermata l'esclusione, con motivazione sostituita.

Motivo: nei laboratori KNN compare una sola volta, come una delle
alternative suggerite per un esercizio di classificazione dopo riduzione
con PCA, e non compare in nessuna forma regressiva. La consegna richiede
il confronto dei modelli di regressione visti nel corso, e KNN per la
regressione non è tra questi.

Cade la motivazione precedente, che aggiungeva la ridondanza rispetto ad
alberi ed ensemble: sotto un obbligo di copertura, la ridondanza non è un
criterio ammissibile per escludere un modello. Vale soltanto
l'appartenenza al programma.

### Tecniche fuori dal materiale del corso

Vanno segnalate come tali ovunque compaiano: la cross-validation con
vincolo di gruppo, che è la trasposizione del K-Fold a dati raggruppati e
che la struttura del dataset rende obbligatoria; le implementazioni
alternative di gradient boosting; i metodi di combinazione di modelli
eterogenei; i modelli che trattano esplicitamente la struttura
sequenziale delle traiettorie.

## [26-08-2026] — Modulo di caricamento e verifica di integrità dei dati grezzi

Il primo codice del progetto è il modulo che legge i file grezzi, insieme allo
script che ne verifica l'integrità.

### Modulo di caricamento

`src/data.py` è l'unico punto della repository in cui i file di `data/raw/`
vengono aperti. Restituisce DataFrame con colonne nominate e tipizzate e una
struttura `CmapssSubset` che tiene insieme training, test ed etichette RUL di uno
stesso sottoinsieme.

Scelte di implementazione e relative motivazioni:

- La lettura avviene senza passare i nomi delle colonne, che vengono assegnati
  solo dopo la verifica che il file ne contenga 26. Motivo: passando i nomi in
  fase di lettura, un file con un numero di campi diverso verrebbe adattato
  silenziosamente invece di far fallire il caricamento.
- Le colonne interamente vuote vengono rimosse prima del controllo. Motivo: le
  righe dei file originali terminano con spazi e, a seconda della versione di
  pandas, questo produce una colonna finale spuria.
- `unit` e `cycle` sono tipizzati come interi. Motivo: sono conteggi, e lasciarli
  in virgola mobile renderebbe fragili i raggruppamenti per unità, su cui si
  regge il vincolo di partizionamento per motore.
- La radice della repository è ricavata dalla posizione del file e non dalla
  directory di lavoro. Motivo: il caricamento deve funzionare in modo identico da
  uno script lanciato dalla radice e da un notebook che risiede in `notebooks/`.
- I sensori sono numerati per posizione (`sensor_01` ... `sensor_21`). Motivo: la
  numerazione è verificabile direttamente sul file, mentre la corrispondenza con
  le sigle fisiche dipende da una fonte esterna al dato.

Gli script si invocano come moduli (`python -m scripts.nome`) e non per percorso.
Motivo: l'invocazione per percorso colloca `scripts/` in cima al percorso di
ricerca di Python anziché la radice, e l'import di `src` fallisce. La soluzione
alternativa, cioè manipolare `sys.path` dentro ogni script, è una toppa che si
propagherebbe a tutti gli script successivi.

### Verifica di integrità

`scripts/verify_raw_data.py` controlla numero di righe e di unità di ogni file
contro valori attesi cablati nel codice, assenza di valori mancanti, contiguità
degli identificativi delle unità, consecutività dei numeri di ciclo entro ogni
unità, corrispondenza tra unità di test ed etichette RUL, positività delle
etichette.

Il controllo sulla consecutività dei cicli è il più importante: se una traiettoria
avesse cicli mancanti, la RUL costruita per differenza dall'ultimo ciclo sarebbe
sbagliata senza che nulla lo segnali.

I valori attesi sono pin di integrità e non parametri: uno scostamento indica
un'acquisizione diversa da quella su cui il progetto è costruito.

ESITO: tutti i controlli superati sui quattro sottoinsiemi.

### Discrepanza con la documentazione ufficiale

Il readme distribuito con il dataset attribuisce a FD004 248 traiettorie di
training e 249 di test. I file contengono l'opposto: 249 unità di training (61249
righe) e 248 di test (41214 righe). La stessa inversione compare nelle fonti
secondarie che ricopiano la tabella del readme.

Il conteggio adottato è quello ricavato dai file. Motivo: la fonte primaria è il
dato, non la documentazione che lo accompagna.

## [26-08-2026] — Esplorazione, perimetro sperimentale e definizione del target

### Artefatti prodotti

`src/explore.py` calcola le statistiche descrittive, `scripts/run_exploration.py`
le salva come otto file CSV in `experiments/exploration/`, e
`notebooks/01_esplorazione.ipynb` li legge producendo otto figure in
`results/figures/` e la tabella riassuntiva in `results/tables/`.

La separazione risponde a un criterio: il notebook non apre i dati grezzi e non
esegue calcoli, quindi si esegue in pochi secondi e non può divergere dagli
artefatti registrati. `experiments/` non è versionato perché rigenerabile,
`results/` sì perché è la traccia verificabile di ciò che è stato osservato.

### Difetto nel criterio di individuazione delle variabili costanti

Sintomo: la libreria di calcolo numerico ha emesso avvisi di correlazione non
definita su FD001 e FD003, e l'elenco delle variabili costanti risultava
incoerente con il conteggio dei valori distinti riportato accanto (due variabili
con un solo valore distinto non comparivano tra le costanti).

Causa: il criterio era `deviazione standard uguale a zero`. Su una colonna di
valori identici la deviazione standard calcolata numericamente non è esattamente
nulla ma un residuo di arrotondamento dell'ordine di 1e-13, e il confronto con
zero fallisce.

Soluzione: il criterio è ora il numero di valori distinti, esatto per costruzione.
Le variabili costanti sono inoltre escluse dal calcolo delle correlazioni, il che
elimina gli avvisi alla radice invece di sopprimerli.

L'errore non faceva fallire nulla e produceva un elenco plausibile: le due
variabili mancate sarebbero entrate nei modelli come colonne prive di
informazione.

### Struttura dei quattro sottoinsiemi

| Sottoinsieme | Motori | Durata mediana | Durata min | Durata max | Dev. std | Regimi | Costanti | max abs Pearson |
|---|---|---|---|---|---|---|---|---|
| FD001 | 100 | 199 | 128 | 362 | 46,3 | 1 | 7 | 0,70 |
| FD002 | 260 | 199 | 128 | 378 | 46,8 | 6 | 0 | 0,07 |
| FD003 | 100 | 220,5 | 145 | 525 | 86,5 | 1 | 6 | 0,69 |
| FD004 | 249 | 234 | 128 | 543 | 73,1 | 6 | 0 | 0,08 |

Le variabili costanti su entrambi i sottoinsiemi a regime singolo sono
`setting_3` e i sensori 01, 05, 16, 18, 19. Il sensore 10 è costante su FD001 ma
assume quattro valori su FD003.

Su FD002 e FD004 nessuna variabile risulta costante, ma non per maggiore
informatività: rapportando la deviazione standard misurata dentro un singolo
regime a quella complessiva, su FD002 il rapporto non supera 0,18 per alcun
sensore e per la maggior parte resta sotto 0,06. Fissata la condizione di volo, la
variabilità delle letture quasi scompare. La variabilità osservata accorpando i
sei regimi è quindi dovuta al regime e non al degrado, ed è la ragione per cui la
correlazione marginale con il target si annulla.

### Perimetro sperimentale: FD001 e FD003

Gli esperimenti sono condotti su FD001 e FD003.

Motivo: la coppia tiene fermo il regime di volo e fa variare il solo numero di
modi di guasto. Il pre-processing resta identico sui due sottoinsiemi, il
protocollo è letteralmente lo stesso, e la replica del confronto su due
popolazioni diverse permette di distinguere una conclusione sui modelli da una
conclusione su un singolo dataset. Le due popolazioni sono effettivamente diverse:
FD003 ha traiettorie più lunghe e quasi doppia dispersione.

Alternative scartate:

- Solo FD001. Un solo sottoinsieme non consente di verificare se la graduatoria
  dei modelli sia stabile, e l'unica motivazione dell'esclusione degli altri
  sarebbe il costo.
- Tutti e quattro come problemi separati. Scartata per il costo: con traiettorie
  da 54000 e 61000 righe, i modelli il cui costo di addestramento cresce più che
  linearmente nel numero di righe (macchine a vettori di supporto, modelli
  additivi generalizzati, selezione esaustiva dei sottoinsiemi di variabili)
  diventerebbero il collo di bottiglia. Il rischio non è la durata degli
  esperimenti ma la pressione a escludere modelli dal confronto, che è esattamente
  ciò che la consegna vieta. In più raddoppierebbe il lavoro di commento, che ha
  lo stesso peso del confronto numerico.
- FD001 e FD002. FD002 ha una distribuzione delle durate praticamente identica a
  FD001, quindi la replica sarebbe meno informativa rispetto a FD003.
- Unione dei quattro in un unico insieme. Mescola popolazioni con regimi e modi di
  guasto diversi e rende impossibile qualunque affermazione sulla difficoltà
  differenziale.

Limite dichiarato: il lavoro non copre il caso a condizioni operative multiple, e
le conclusioni valgono per il regime singolo. Rendere utilizzabili FD002 e FD004
richiederebbe uno stadio di normalizzazione dei sensori entro regime, che
sposterebbe il baricentro del lavoro dalla comparazione tra modelli alla
progettazione del pre-processing.

### Previsione smentita sulla collinearità

L'aspettativa iniziale era che i sensori più correlati con il target fossero anche
fortemente correlati tra loro, e che la dimensionalità effettiva fosse molto minore
di 21. La misura la smentisce: su FD001 una sola coppia su 105 supera 0,9 in
valore assoluto (`sensor_09` e `sensor_14`, 0,963), su FD003 tre coppie su 120.

Conseguenza per il seguito: su questi dati la giustificazione della
regolarizzazione e della regressione sulle componenti principali non può poggiare
sulla ridondanza tra variabili esplicative, che è modesta. Va motivata sul
rapporto tra numero di variabili e numero di unità indipendenti: le righe sono
decine di migliaia, ma i motori sono cento, e la numerosità campionaria rilevante
è la seconda.

### Definizione del target

Il target è la vita utile residua, ottenuta sulle traiettorie di training come
differenza tra il ciclo del guasto e il ciclo corrente. Sulle traiettorie di test,
troncate prima del guasto, la stessa quantità si ricava dal file di etichette:
`src/target.py` implementa le due strade separatamente.

Sul target è applicata una censura a soglia, con soglia fissata a 125 cicli.

Motivo: nella prima parte della vita di un motore il degrado non è osservabile dai
sensori, e le letture di unità con vite residue molto diverse sono in quella fase
indistinguibili. Un target lineare chiede di predire valori diversi a partire da
ingressi uguali, e questa componente irriducibile pesa in modo sproporzionato in
una metrica quadratica perché ricade sui valori più grandi.

La soglia di 125 è inferiore alla durata della traiettoria più breve di entrambi i
sottoinsiemi in perimetro (128 cicli in FD001, 145 in FD003), quindi ogni motore
attraversa sia la fase censurata sia la fase di degrado e nessuna traiettoria
risulta interamente costante.

Alternative scartate:

- Target lineare non censurato. Introduce nella metrica una componente che nessun
  modello può ridurre, indebolendo proprio il confronto tra modelli.
- Soglia scelta per cross-validation. Non è un iperparametro ma parte della
  definizione del problema: cambiando la soglia cambia la scala del target, e
  abbassandola l'errore quadratico medio cala per costruzione. Un confronto tra
  soglie basato sull'errore selezionerebbe sempre la più bassa.
- Confronto completo su entrambe le definizioni. Scartata per il costo, con lo
  stesso ragionamento applicato ai sottoinsiemi.

È previsto un controllo di sensibilità: a fine lavoro il modello risultato migliore
e la baseline lineare regolarizzata vengono rieseguiti anche con target non
censurato. Sono due addestramenti aggiuntivi, e rendono verificabile se la
graduatoria dipenda dalla soglia.

### Verifica dell'assunzione su cui poggia la censura

La censura assume che nella fase iniziale di vita le letture non varino al variare
della vita residua. L'assunzione è misurata calcolando la correlazione tra sensori
e vita utile residua separatamente sopra e sotto la soglia.

| Sottoinsieme | Sensore | Pearson oltre soglia | Pearson entro soglia |
|---|---|---|---|
| FD001 | sensor_11 | -0,17 | -0,77 |
| FD001 | sensor_04 | -0,16 | -0,74 |
| FD001 | sensor_12 | +0,16 | +0,74 |
| FD001 | sensor_07 | +0,14 | +0,72 |
| FD001 | sensor_15 | -0,14 | -0,71 |
| FD003 | sensor_11 | -0,32 | -0,78 |
| FD003 | sensor_04 | -0,28 | -0,73 |
| FD003 | sensor_13 | -0,42 | -0,69 |
| FD003 | sensor_08 | -0,42 | -0,69 |
| FD003 | sensor_17 | -0,33 | -0,70 |

Su FD001 la separazione è netta e l'assunzione regge: la censura è coerente con la
struttura del dato e non solo con una convenzione.

Su FD003 la separazione esiste ma è meno pronunciata: sopra soglia le correlazioni
raggiungono 0,42, quindi una parte di informazione utile è presente già prima
della soglia e la censura la scarta. La lettura plausibile è che con due modi di
guasto una parte della popolazione degradi più precocemente, ma resta
un'interpretazione.

Limite dichiarato: la soglia è un'ipotesi di modellazione e non una quantità
misurata, i valori assoluti delle metriche dipendono da essa, e su FD003 scarta una
parte di segnale. La soglia non è stata modificata dopo questa misura: era fissata
a priori, e cambiarla dopo averla vista sarebbe una scelta fatta guardando il
risultato.

Su FD002 e FD004 le correlazioni sopra soglia sono nulle e sotto soglia raggiungono
appena 0,10, il che conferma per via indipendente che in quei sottoinsiemi il
segnale è schiacciato dal regime operativo e non dalla fase di vita.