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

## [26-08-2026] — Protocollo di valutazione: decisioni, implementazione e convalida

Il protocollo con cui tutti i modelli verranno confrontati è definito e
implementato prima che venga addestrato un solo modello del confronto. Le
quattro decisioni che lo compongono sono registrate qui insieme all'esito della
convalida della catena.

### Ruolo dei file di verifica ufficiali

I file `test_FD00X.txt` e `RUL_FD00X.txt` costituiscono l'insieme di verifica
finale. Non entrano in nessuna scelta (variabili, iperparametri, graduatoria) e
vengono letti una sola volta, a graduatoria chiusa, su ciascun modello già
selezionato e riaddestrato sull'intera parte di addestramento.

Le traiettorie di verifica sono troncate e ogni unità ha una sola etichetta,
riferita all'ultimo ciclo osservato. Da quella si ricava il target a ogni ciclo
sommando i cicli che mancano alla fine della traiettoria osservata, quindi la
parte di verifica è utilizzabile per intero. Sono adottate tre letture: su tutti
i cicli, sul solo ultimo ciclo di ciascuna unità (forma con cui il dataset è
riportato in letteratura), e sull'ultimo ciclo contro il target non censurato
(variante in cui la censura si applica all'addestramento ma non alla verifica).
Le tre hanno costo nullo l'una rispetto all'altra: sono sottoinsiemi e varianti
di confronto delle stesse predizioni.

Alternative scartate:

- Non usare i file di verifica e ricavare l'insieme finale dalle sole
  traiettorie di addestramento. Scarta un insieme indipendente già disponibile e
  di dimensione pari all'addestramento (100 motori per sottoinsieme), e priva il
  lavoro dell'unica stima non condizionata dalla selezione.
- Usare i soli ultimi cicli. Riduce la verifica a 100 punti per sottoinsieme e
  rinuncia alle restanti 13.000 e 16.500 righe, che sono corredate di target
  ricostruibile.
- Unire verifica e addestramento e ripartizionare. Le traiettorie di verifica
  sono troncate e non consentono di costruire il target per differenza:
  l'unione richiederebbe due definizioni diverse della stessa variabile dentro
  lo stesso insieme, e distruggerebbe un insieme di verifica già dato.

### Schema di cross-validation

Cross-validation non annidata, K-Fold con vincolo di gruppo sul motore, 5 fold,
ripetuta su 3 semi.

Il numero di fold segue dalla numerosità: 100 motori per sottoinsieme, quindi 20
motori e circa 4.000 righe per parte di verifica. Dieci fold porterebbero la
verifica a 10 motori, rendendo instabile la stima del singolo fold; il
leave-one-group-out porterebbe a 100 addestramenti per configurazione, costo non
sostenibile sui modelli il cui addestramento cresce più che linearmente nel
numero di righe.

La conduzione è in due stadi. La ricerca su griglia degli iperparametri opera sui
5 fold del seme 0, un solo passaggio. La configurazione selezionata viene poi
rivalutata su 5 fold per 3 semi, e i 15 punteggi risultanti producono la media e
la deviazione standard che entrano nella tabella di confronto. Motivo dello
sdoppiamento: la ripetizione dentro la ricerca triplicherebbe il costo di ogni
griglia, che sulle macchine a vettori di supporto (circa 16.500 righe di
addestramento per fold) è la differenza fra un esperimento eseguibile e uno che
non lo è.

La cross-validation non è annidata: le stesse partizioni servono a scegliere gli
iperparametri e a riportare il punteggio della configurazione scelta, come nei
laboratori del corso. Il punteggio riportato è quindi ottimisticamente distorto.
La scelta è ammissibile perché esiste un insieme di verifica esterno che non
partecipa alla selezione: senza di esso la cross-validation annidata sarebbe
obbligatoria. Alternativa scartata: cross-validation annidata, che moltiplica il
costo per il numero di fold esterni e produce iperparametri diversi in ciascun
fold esterno, il che rende impossibile identificare un modello selezionato di cui
leggere coefficienti e importanze, cioè il materiale con cui si costruisce il
commento di ciascun modello.

La deviazione standard riportata è calcolata sui fold e non è l'errore standard
della media: i fold condividono le righe di addestramento e non sono
indipendenti. È una misura di dispersione e come tale va letta. Due modelli il
cui divario è inferiore alla dispersione fra fold si considerano non
distinguibili sotto questo protocollo e non vengono ordinati.

Lo schema con vincolo di gruppo non compare nei laboratori del corso: è la
trasposizione diretta del K-Fold a dati raggruppati, resa obbligatoria dalla
struttura del dataset.

### Metrica di riferimento

La metrica su cui si selezionano gli iperparametri e si ordina la graduatoria è
la radice dell'errore quadratico medio. Motivo: è nelle unità del target
(cicli), quindi interpretabile e commentabile; è coerente con la perdita
minimizzata dalla maggior parte dei modelli in confronto, quindi non introduce
disallineamento fra ciò che i modelli ottimizzano e ciò su cui vengono giudicati.

A corredo sono riportati, e mai usati per selezionare, l'errore assoluto medio e
il coefficiente di determinazione. Il rapporto fra le prime due dice se l'errore
è dominato da una coda di errori grandi o è diffuso, il che su questo dataset è
informativo perché la censura crea una fase a target costante e una fase finale
di degrado con profili di errore diversi. Il terzo è adimensionale e serve a
confrontare sottoinsiemi con varianza del target diversa. L'errore quadratico
medio non è riportato separatamente perché è il quadrato della metrica di
riferimento.

Le metriche sono calcolate per fold e poi mediate, non aggregando in un unico
vettore le predizioni di tutti i fold: l'aggregazione produrrebbe un numero solo
e perderebbe la dispersione, che è parte del risultato.

Alternativa scartata: la funzione di punteggio asimmetrica adottata in
letteratura su C-MAPSS, che penalizza più severamente le predizioni tardive.
Scartata perché fuori dal materiale del corso e perché l'asimmetria è
un'assunzione di dominio sulla gravità relativa dei due tipi di errore, che il
lavoro non è in grado di giustificare.

### Rappresentazione delle osservazioni

Una riga per ciclo, letture grezze dei sensori, nessuna aggregazione su finestre
temporali.

Alternativa scartata: aggiunta di aggregazioni su finestra mobile (medie e
deviazioni standard degli ultimi cicli, scostamenti, pendenze locali). È
l'intervento che su questi dati produce il guadagno maggiore, perché attenua il
rumore di misura e rende visibile la deriva. È però ingegnerizzazione di
variabili su serie temporali, fuori dal materiale del corso; introduce
l'ampiezza della finestra come iperparametro aggiuntivo da selezionare in
validazione, moltiplicando il costo di ogni griglia; e richiede che la finestra
sia strettamente causale, con un rischio concreto di fuga di informazione nei
primi cicli di ogni traiettoria. Soprattutto sposterebbe il baricentro del lavoro
dalla comparazione fra modelli alla progettazione delle variabili.

Alternativa scartata: una riga per finestra con le sole aggregazioni. Cambia
l'unità di osservazione e rende non confrontabile la lettura sull'ultimo ciclo
delle traiettorie di verifica.

Il numero di ciclo è incluso fra le variabili esplicative. Non è una fuga di
informazione: il numero di cicli percorsi è noto al momento della predizione
anche su una traiettoria troncata. Va però tenuto presente che sulle traiettorie
complete la vita utile residua è per costruzione la differenza fra durata e ciclo
corrente, mentre su quelle troncate il punto di interruzione è casuale: la
relazione non si trasferisce integralmente dall'addestramento alla verifica.

Per rendere misurabile questa componente la tabella dei risultati è preceduta da
due baseline: la predizione costante pari alla media del target di addestramento,
che è il pavimento assoluto, e la regressione sul solo numero di ciclo, che è il
pavimento informativo. Il guadagno di ciascun modello si legge rispetto alla
seconda.

Le colonne costanti sono rimosse, con criterio basato sul numero di valori
distinti e applicato alle sole traiettorie di addestramento del sottoinsieme. Il
criterio è identico sui due sottoinsiemi e produce liste diverse: `sensor_10` è
costante su FD001 e assume quattro valori su FD003, quindi viene rimosso solo dal
primo. La costanza è una proprietà strutturale del sensore in quel regime
operativo e non dipende dal target: determinarla sull'intera parte di
addestramento non introduce informazione proveniente dalle partizioni di
verifica. `setting_1` e `setting_2` sono mantenute: non sono costanti, e il loro
contributo nullo è materiale per il commento dei modelli con selezione delle
variabili.

La standardizzazione è applicata dentro la pipeline a tutti i modelli, anche a
quelli per cui è irrilevante. Un pre-processing differenziato per famiglia
introdurrebbe una differenza di condizioni fra modelli confrontati, che è
esattamente ciò che il protocollo deve escludere.

Matrice risultante: 18 variabili su FD001, 19 su FD003.

### Implementazione

`src/protocol.py` contiene lo schema di partizionamento, il numero di fold, i
semi, le metriche e le funzioni di valutazione. È l'unico punto in cui queste
quantità sono scritte, e ogni esperimento vi passa attraverso: è così che il
confronto a parità di condizioni è garantito dal codice e non dalla disciplina di
chi lo usa. Le funzioni accettano qualunque oggetto con `fit` e `predict`, quindi
nessun modello può ricevere un trattamento diverso dagli altri.

`src/pipeline.py` compone selezione delle colonne, standardizzazione e modello.
`src/design.py` costruisce la matrice di progetto e le tre letture della parte di
verifica. `src/baselines.py` fornisce le due baseline.
`scripts/run_protocol_check.py` esercita la catena.

Lo stimatore viene clonato prima di ogni addestramento. Senza clonazione un
oggetto già adattato e riaddestrato su un altro fold può conservare stato, il che
non fa fallire nulla e produce numeri leggermente sbagliati.

### Controlli di correttezza superati

Quattro controlli che potevano fallire.

Coerenza fra target e metrica: la radice dell'errore quadratico medio della
predizione costante vale 41,694 su FD001 contro una deviazione standard del
target di 41,674, e 40,730 su FD003 contro 40,627. Lo scarto residuo è dovuto al
fatto che la costante è la media dei motori di addestramento del fold e non
quella del fold di verifica, ed è anche la ragione del coefficiente di
determinazione lievemente negativo.

Assenza di sovrapposizione: nelle 15 partizioni di ciascun sottoinsieme nessun
motore compare contemporaneamente in addestramento e in verifica.

Integrità del target di verifica: il target ricostruito dalle etichette coincide,
sull'ultimo ciclo di ogni unità, con l'etichetta stessa censurata alla soglia. Il
controllo è dentro la costruzione della matrice e fa fallire il caricamento. È il
controllo più importante della catena: un disallineamento posizionale fra
etichette e unità, o una costruzione del target di verifica per differenza
anziché dalle etichette, produrrebbe un target quasi costante e facile da
predire, cioè un risultato migliore del vero senza che nulla lo segnali.

Coerenza con l'esplorazione: la baseline sul solo numero di ciclo ottiene un
coefficiente di determinazione di 0,549 su FD001 e 0,244 su FD003. Il divario è
la conseguenza diretta della dispersione delle durate misurata in fase di
esplorazione (46,3 contro 86,5): un conteggio dei cicli è tanto meno informativo
quanto più le durate variano. Due misure indipendenti che si spiegano a vicenda.

### Effetto del vincolo di gruppo

Confronto diagnostico a parità di modello, numero di fold e seme, con l'unica
differenza del vincolo di gruppo. I due modelli impiegati non sono ottimizzati e
non appartengono al confronto.

| Sottoinsieme | Modello | Per riga | Per unità | Ottimismo | Relativo |
|---|---|---|---|---|---|
| FD001 | Regressione lineare | 19,98 | 20,33 | 0,35 | 1,7 % |
| FD001 | Foresta casuale | 15,89 | 16,73 | 0,85 | 5,1 % |
| FD003 | Regressione lineare | 19,27 | 20,05 | 0,79 | 3,9 % |
| FD003 | Foresta casuale | 13,10 | 15,26 | 2,17 | 14,2 % |

L'effetto è reale e sistematico, e cresce con la capacità del modello di
memorizzare le righe vicine e con la lunghezza delle traiettorie: FD003 ha
traiettorie più lunghe, quindi più cicli quasi identici per motore, e vi si
osserva l'ottimismo maggiore.

L'effetto è però più contenuto di quanto la motivazione qualitativa lasciasse
prevedere. Su FD001 un partizionamento per riga sottostimerebbe l'errore di poco
più del 5 per cento anche su un modello a capacità alta. Due cautele nella
lettura: i modelli diagnostici non sono ottimizzati, e sotto partizione per riga
anche la selezione degli iperparametri deriverebbe, aggiungendo un ottimismo che
questa misura non cattura. La tabella mostra che il vincolo di gruppo sposta i
margini del confronto, non che senza di esso i risultati sarebbero privi di
significato.

### Convalida sulla regressione lineare

Cross-validation per unità motore, 5 fold per 3 semi, media e deviazione standard
sui 15 addestramenti.

| Sottoinsieme | Modello | RMSE | MAE | R² |
|---|---|---|---|---|
| FD001 | Predizione costante | 41,69 ± 0,14 | 36,98 ± 0,12 | -0,002 ± 0,003 |
| FD001 | Solo numero di ciclo | 27,88 ± 2,47 | 21,49 ± 1,71 | 0,549 ± 0,080 |
| FD001 | Regressione lineare | 20,35 ± 1,18 | 16,50 ± 1,03 | 0,761 ± 0,028 |
| FD001 | Regressione lineare senza numero di ciclo | 21,68 ± 1,30 | 17,75 ± 1,06 | 0,728 ± 0,032 |
| FD003 | Predizione costante | 40,73 ± 0,65 | 35,55 ± 0,14 | -0,007 ± 0,008 |
| FD003 | Solo numero di ciclo | 35,12 ± 2,64 | 27,53 ± 2,12 | 0,244 ± 0,147 |
| FD003 | Regressione lineare | 19,93 ± 1,46 | 15,62 ± 1,29 | 0,757 ± 0,037 |
| FD003 | Regressione lineare senza numero di ciclo | 19,85 ± 1,45 | 15,50 ± 1,26 | 0,760 ± 0,033 |

Insieme di verifica ufficiale, riaddestramento sui 100 motori e lettura unica.

| Sottoinsieme | Modello | RMSE tutti i cicli | R² tutti i cicli | RMSE ultimo ciclo | R² ultimo ciclo |
|---|---|---|---|---|---|
| FD001 | Predizione costante | 35,34 | -0,642 | 41,94 | -0,095 |
| FD001 | Solo numero di ciclo | 23,69 | 0,262 | 32,25 | 0,352 |
| FD001 | Regressione lineare | 19,07 | 0,522 | 21,45 | 0,714 |
| FD001 | Regressione lineare senza numero di ciclo | 20,75 | 0,434 | 20,83 | 0,730 |
| FD003 | Predizione costante | 31,37 | -0,596 | 43,70 | -0,245 |
| FD003 | Solo numero di ciclo | 26,03 | -0,099 | 36,80 | 0,117 |
| FD003 | Regressione lineare | 17,96 | 0,477 | 21,44 | 0,700 |
| FD003 | Regressione lineare senza numero di ciclo | 17,96 | 0,477 | 21,16 | 0,708 |

Le letture contro target non censurato differiscono da quelle censurate di circa
un ciclo sulla radice dell'errore quadratico medio, quantità coerente con il
numero ridotto di unità di verifica la cui vita residua supera la soglia.

### Incomparabilità fra le letture

L'errore assoluto sull'insieme di verifica risulta inferiore a quello in
cross-validation (19,07 contro 20,35 su FD001, 17,96 contro 19,93 su FD003). Non
è una fuga di informazione: è un effetto della composizione delle due
popolazioni. Le traiettorie di verifica sono troncate in un punto casuale prima
del guasto e contengono quindi in proporzione molte più righe della fase iniziale
di vita, dove il target è appiattito sulla soglia. La quota di righe al valore di
soglia passa dal 39,4 al 61,4 per cento su FD001 e dal 49,4 al 69,1 per cento su
FD003, e la deviazione standard del target scende da 41,67 a 27,58 e da 40,63 a
24,84. Il target da predire varia meno, e l'errore assoluto cala per costruzione.

Il coefficiente di determinazione si muove nella direzione opposta, da 0,761 a
0,522 e da 0,757 a 0,477: rispetto alla variabilità disponibile la prestazione
sulla verifica è peggiore, che è la direzione attesa.

Anche il coefficiente di determinazione, però, non è confrontabile fra le due
letture, perché ha denominatori diversi. Lo stesso vale fra le due letture della
verifica: sull'ultimo ciclo il target è meno censurato e più disperso, e il
coefficiente sale a 0,714 e 0,700 pur essendo la radice dell'errore quadratico
medio più alta che su tutti i cicli.

Ne consegue una regola di lettura che vale per l'intera tabella dei risultati:
cross-validation, verifica su tutti i cicli e verifica sull'ultimo ciclo sono tre
letture su popolazioni diverse, e i loro valori non si sottraggono fra loro. Ciò
che si confronta legittimamente è la graduatoria dei modelli dentro ciascuna
lettura, e la funzione dell'insieme di verifica è mostrare se quella graduatoria
si conservi su una popolazione indipendente.

### Contributo del numero di ciclo

L'aspettativa era che il numero di ciclo apportasse una quota rilevante della
capacità predittiva, tale da appiattire il confronto fra modelli. La misura la
ridimensiona.

In cross-validation la sua rimozione peggiora la radice dell'errore quadratico
medio di 1,33 cicli su FD001, quantità confrontabile con la dispersione fra fold
(1,18 e 1,30), e la migliora di 0,09 cicli su FD003, cioè non produce alcun
effetto. Sull'ultimo ciclo dell'insieme di verifica la rimozione migliora il
risultato su entrambi i sottoinsiemi (20,83 contro 21,45 su FD001, 21,16 contro
21,44 su FD003).

La lettura è coerente con la struttura del dato: sulle traiettorie complete il
numero di ciclo è legato al target da una relazione esatta, su quelle troncate
no, perché il punto di interruzione è casuale. Sulle traiettorie di verifica
lette per intero la variabile aiuta comunque, perché i cicli iniziali
corrispondono a vite residue alte e quindi censurate; sull'ultimo ciclo, dove il
troncamento agisce, è lievemente fuorviante.

La decisione di includerlo resta invariata: il numero di cicli percorsi è
informazione realmente disponibile al momento della predizione, ed escluderla
perché su una delle letture peggiora leggermente il risultato sarebbe una scelta
fatta guardando l'esito. La baseline sul solo numero di ciclo resta in tabella
come termine di lettura.

### Limiti dichiarati

Il punteggio riportato in cross-validation è ottimisticamente distorto, perché le
stesse partizioni servono a selezionare gli iperparametri e a riportarne
l'esito. Il divario con la lettura sull'insieme di verifica non ne è una misura
diretta, per l'incomparabilità delle popolazioni descritta sopra.

La dispersione fra fold non è un errore standard e non consente test di
significatività: i fold condividono le righe di addestramento.

La rappresentazione a letture grezze non sfrutta la struttura temporale delle
traiettorie, e i valori assoluti delle metriche restano perciò distanti da quelli
ottenibili con variabili aggregate su finestra.

La predizione costante ottiene un coefficiente di determinazione marcatamente
negativo sull'insieme di verifica (-0,642 e -0,596 su tutti i cicli) perché la
media del target di addestramento (86,8 e 93,1) è distante da quella della
verifica (108,9 e 112,3). È un'ulteriore manifestazione della differenza di
composizione fra le due popolazioni.

ESITO: protocollo definito, implementato e convalidato end to end sulla
regressione lineare e sulle due baseline, su entrambi i sottoinsiemi in
perimetro. Nessun modello del confronto è stato addestrato.

## [26-08-2026] — Blocco lineare: infrastruttura di esperimento, metodi di ricampionamento, modelli lineari e selezione delle variabili

### Infrastruttura di esperimento

Il confronto fra modelli richiede che ogni blocco sia valutato sotto la stessa
procedura. La composizione dei due stadi previsti dal protocollo (ricerca della
configurazione sulle 5 partizioni del seme 0, rivalutazione della sola
configurazione selezionata sulle 15 partizioni dei tre semi) è stata quindi
scritta in un unico punto riusabile, invece di essere ripetuta negli script dei
singoli blocchi.

Moduli aggiunti: `src/registry.py` (stimatori, griglie e funzioni di lettura dei
parametri di ciascun modello), `src/search.py` (ricerca su griglia con le tre
metriche e controllo sui bordi), `src/experiment.py` (motore a due stadi,
tabella di confronto, percorsi dei coefficienti, controllo diagnostico con
selezione annidata), `src/selection.py` (motore di stima e tre metodi di
selezione delle variabili), `src/resampling.py` (procedure di stima dell'errore
del laboratorio 6). Script: `scripts/run_linear_models.py`,
`scripts/run_resampling.py`, `scripts/run_selection_check.py`.

Motivo della separazione fra registro e motore: i blocchi successivi aggiungono
il proprio registro senza toccare il motore, quindi nessun modello può ricevere
un trattamento diverso dagli altri per effetto di codice duplicato e divergente.

### Best subset selection: motore di stima dedicato

Nella forma del laboratorio 7 la ricerca esaustiva costruisce e valuta una
pipeline per ciascun sottoinsieme, il che richiede 262.143 stime per fold su
FD001 e 524.287 su FD003 e non è eseguibile.

Alternative valutate. Ricerca esaustiva limitata a una cardinalità massima di 4:
scartata perché il limite sarebbe arbitrario e produrrebbe un modello
confrontato con forward stepwise a parità di nome ma non di spazio di ricerca.
Ricerca esaustiva su un pool ridotto di variabili, come nel laboratorio, che ne
usa 8 su 10: scartata perché qui la scelta del pool richiederebbe una
preselezione supervisionata fuori dal flusso di validazione.

Soluzione adottata: i minimi quadrati su un sottoinsieme si ottengono dalle
sottomatrici di X'X e X'y, che dipendono dalla partizione e non dal
sottoinsieme, e l'errore sulla parte di verifica si scrive come forma quadratica
nei coefficienti senza costruire le predizioni. Il costo per sottoinsieme passa
dall'ordine del numero di righe a quello del quadrato del numero di variabili
selezionate. La ricerca esaustiva completa è risultata eseguibile in 11,0 s su
FD001 e 22,4 s su FD003.

La riformulazione è algebricamente esatta ma è codice del progetto e non di
libreria, quindi un errore avrebbe prodotto numeri plausibili e sbagliati.
`scripts/run_selection_check.py` verifica tre proprietà: coincidenza con la
valutazione ordinaria sotto `src.protocol` su sottoinsiemi casuali, coincidenza
della ricerca esaustiva veloce con quella ingenua su un pool di 8 variabili,
impossibilità che una ricerca greedy batta l'esaustiva a parità di cardinalità.

ESITO: tutti i controlli superati. Scarto massimo 7·10⁻¹⁵ su dati generati e
3,8·10⁻¹¹ sulla matrice di FD001, contro una tolleranza di 10⁻⁸. La crescita di
tre ordini di grandezza fra dati generati e dati reali è dovuta al
condizionamento della matrice nel sistema normale e resta cinque ordini di
grandezza sotto la soglia.

### Selezione delle variabili rispetto ai fold

I tre metodi di selezione scelgono un sottoinsieme guardando un punteggio di
cross-validation. Due collocazioni possibili: trattare la selezione come un
iperparametro, cercandola sulle partizioni del seme 0 come per ogni altro
modello, oppure rifarla dentro ogni fold di rivalutazione.

La seconda produce punteggi privi della distorsione della selezione, ma i 15
fold selezionano sottoinsiemi diversi e non identificano un modello di cui
commentare le variabili; inoltre tratterebbe questi tre modelli in modo più
severo degli altri, il che è una disparità di condizioni nel confronto.

Scelta adottata: la prima, coerente con il protocollo già registrato, con
l'aggiunta di un controllo diagnostico che misura quanto costa. Il controllo
rifà la selezione dentro ciascuna delle 15 partizioni, con una
cross-validation interna sui soli motori di addestramento, e valuta il
sottoinsieme risultante sulla parte di verifica che non ha partecipato alla
scelta. Non entra in graduatoria.

### Griglie degli iperparametri

Ridge: `logspace(-2, 8, 41)`. Lasso: `logspace(-4, 4, 50)`, la griglia del
laboratorio. Elastic Net: la stessa griglia di penalizzazione per nove valori
del bilanciamento fra le due penalità. Regressione sulle componenti principali:
griglia completa da una componente al numero di variabili.

L'estensione della griglia di Ridge era motivata dalla diversa
parametrizzazione delle due penalità in scikit-learn: Ridge minimizza la somma
dei quadrati dei residui più la penalità, mentre Lasso ed Elastic Net dividono
la parte di errore per il numero di righe, quindi a parità di valore del
parametro la contrazione su Ridge è più debole di un fattore pari al numero di
righe. La previsione operativa che ne era stata tratta si è rivelata errata: le
configurazioni selezionate sono 1.000 su FD001 e 1.778 su FD003, entrambe
interne all'intervallo del laboratorio. L'estensione è mantenuta perché il suo
costo è nullo e perché documenta che la selezione non è vincolata dall'estremo.

Regola sui bordi, fissata prima di eseguire le ricerche e verificata dal codice:
se la configurazione selezionata cade su un estremo della griglia, la griglia
viene estesa da quel lato e la ricerca rieseguita, e il fatto viene registrato.

CORREZIONE alla griglia di Elastic Net. Nella convalida della catena su griglie
ridotte la configurazione selezionata è caduta sul valore minimo del
bilanciamento, che era 0,1. La griglia è stata estesa verso il basso con i
valori 0,01 e 0,05 prima di eseguire la versione completa. Il limite inferiore
del bilanciamento è Ridge, che compare in tabella come modello a sé, quindi
l'estensione infittisce l'avvicinamento a un modello già presente e non apre
uno spazio nuovo. Il valore nullo non è incluso perché coinciderebbe con Ridge
stimato per discesa coordinata anziché in forma chiusa. Dopo l'estensione la
configurazione selezionata è 0,05 su entrambi i sottoinsiemi, quindi interna
alla griglia.

Sulla regressione sulle componenti principali la configurazione selezionata su
FD003 è il numero massimo di componenti. La regola sui bordi non si applica:
oltre il numero di variabili non esistono componenti, quindi il bordo è il
limite strutturale della tecnica e non un vincolo di griglia. Con tutte le
componenti la trasformazione è una rotazione della matrice, e infatti il modello
coincide numericamente con la regressione lineare multipla (19,934353 in
entrambi i casi). La coincidenza vale anche come controllo di correttezza
dell'implementazione.

### Problema tecnico nella tabella di confronto

Sintomo: nella prima esecuzione la colonna che riporta il divario dalla riga
migliore in unità di dispersione non seguiva l'ordinamento dell'errore, e un
modello peggiore risultava più vicino di uno migliore. Causa radice: ogni
divario era diviso per la dispersione della propria riga, quindi un modello più
stabile risultava più vicino a parità di divario. Soluzione: la scala combina la
dispersione della riga e quella della riga migliore. La quantità resta una
scala di lettura e non consente test di significatività, perché i fold
condividono le righe di addestramento.

### Metodi di ricampionamento del laboratorio 6

Le quattro procedure di stima dell'errore sono state applicate a un unico
modello, la regressione lineare multipla, ricampionando le unità motore e non
le righe, per la stessa ragione per cui il partizionamento del protocollo
avviene per motore. L'esclusione di una osservazione per volta diventa quindi
esclusione di un motore per volta. Il bootstrap è implementato come funzione di
ricampionamento scritta da zero, con la firma richiesta dall'esercizio, ed è
usato in due modi: distribuzione dei coefficienti su 200 ricampionamenti e
stima dell'errore sui motori mai estratti.

| Procedura | FD001 | FD003 | stime |
|---|---|---|---|
| Partizione unica (20 semi) | 20,29 ± 1,50 | 19,56 ± 1,13 | 20 |
| Un motore per volta | 19,30 ± 6,52 | 19,57 ± 6,33 | 100 |
| K-Fold a 5 | 20,35 ± 1,18 | 19,93 ± 1,46 | 15 |
| K-Fold a 10 | 20,19 ± 2,43 | 19,74 ± 2,52 | 30 |
| Bootstrap sui motori | 20,37 ± 1,05 | 20,00 ± 0,93 | 200 |

Le medie si ordinano secondo la numerosità della parte di addestramento: 99
motori per l'esclusione di un motore per volta, 90 per il K-Fold a 10, 80 per
quello a 5, circa 63 motori distinti per il bootstrap. È il compromesso fra
distorsione e varianza delle procedure di ricampionamento, misurato sui dati
del progetto. L'unica eccezione è la partizione unica su FD003, che con 70
motori produce la stima più bassa, ed è coerente con il fatto che sia la
procedura più rumorosa.

Le dispersioni non misurano la stessa quantità e non sono intercambiabili: per
la partizione unica descrivono la variabilità fra partizioni, per il K-Fold e
per l'esclusione di un motore per volta la variabilità fra parti di verifica di
una stessa procedura, per il bootstrap la variabilità fra campioni. La
dispersione dell'esclusione di un motore per volta è la più ampia perché ogni
stima è calcolata su una singola traiettoria: misura quanto i motori
differiscono fra loro, non l'incertezza della procedura. Per la stessa ragione
il coefficiente di determinazione calcolato su un solo motore non è
interpretabile.

Il K-Fold a 10 ha dispersione doppia rispetto a quello a 5 con media quasi
identica, perché ogni parte di verifica contiene 10 motori invece di 20. È la
giustificazione empirica del numero di fold fissato nel protocollo.

ESITO: la partizione unica, ripetuta su venti semi, produce su FD001 stime che
vanno da 17,28 a 23,05 per lo stesso modello sugli stessi dati, con la sola
differenza di quali motori finiscono da che parte. Il divario che separa tutti i
modelli del blocco lineare è di 0,03 cicli su FD001 e 0,09 su FD003, due ordini
di grandezza sotto quella escursione. Valutare con una partizione unica avrebbe
reso il confronto indistinguibile dal rumore di partizionamento e avrebbe
consentito di proclamare vincitore qualunque modello scegliendo il seme
opportuno.

### Confronto del blocco lineare

Otto modelli più le due baseline, sotto il protocollo del progetto: media e
deviazione standard sulle 15 partizioni.

FD001, 18 variabili

| Modello | Configurazione | RMSE | Divario |
|---|---|---|---|
| Ridge | alpha = 1000 | 20,327 ± 1,137 | 0,00 |
| Elastic Net | alpha = 0,0869, bilanciamento 0,05 | 20,329 ± 1,125 | 0,00 |
| Best subset | k = 17 | 20,345 ± 1,183 | 0,02 |
| Forward stepwise | k = 17 | 20,345 ± 1,183 | 0,02 |
| Backward stepwise | k = 17 | 20,345 ± 1,183 | 0,02 |
| Lasso | alpha = 0,0281 | 20,345 ± 1,182 | 0,02 |
| Regressione lineare multipla | nessun iperparametro | 20,345 ± 1,183 | 0,02 |
| Componenti principali | 6 componenti | 20,353 ± 1,173 | 0,02 |
| Baseline solo numero di ciclo | | 27,878 ± 2,467 | 3,93 |
| Baseline costante | | 41,694 ± 0,135 | 26,39 |

FD003, 19 variabili

| Modello | Configurazione | RMSE | Divario |
|---|---|---|---|
| Forward stepwise | k = 15 | 19,849 ± 1,449 | 0,00 |
| Best subset | k = 15 | 19,849 ± 1,449 | 0,00 |
| Backward stepwise | k = 15 | 19,849 ± 1,449 | 0,00 |
| Ridge | alpha = 1778 | 19,882 ± 1,449 | 0,02 |
| Elastic Net | alpha = 0,1265, bilanciamento 0,05 | 19,890 ± 1,454 | 0,03 |
| Lasso | alpha = 0,1842 | 19,919 ± 1,459 | 0,05 |
| Componenti principali | 19 componenti | 19,934 ± 1,463 | 0,06 |
| Regressione lineare multipla | nessun iperparametro | 19,934 ± 1,463 | 0,06 |
| Baseline solo numero di ciclo | | 35,116 ± 2,645 | 7,16 |
| Baseline costante | | 40,730 ± 0,651 | 18,59 |

Il divario è espresso in unità di dispersione fra fold. Tutti i modelli del
blocco cadono entro 0,06 dispersioni su entrambi i sottoinsiemi: sotto questo
protocollo non sono distinguibili e non vengono ordinati. La graduatoria resta
aperta. La distanza dalle baseline è invece leggibile, ed è la sola differenza
di queste tabelle che vada interpretata come reale.

Il pareggio ha una spiegazione strutturale coerente con quanto misurato in
esplorazione: le righe sono tre ordini di grandezza più numerose delle
variabili e la correlazione massima fra sensori è 0,963, quindi la stima dei
minimi quadrati non ha varianza in eccesso da ridurre e ogni forma di
contrazione può al più pareggiarla. Il limite dei modelli di questo blocco non
è la varianza della stima ma la forma della relazione fra letture e vita
residua, che possono descrivere solo come combinazione lineare delle letture al
ciclo corrente.

### Selezione delle variabili: risultati

I tre metodi selezionano lo stesso identico sottoinsieme su entrambi i
sottoinsiemi. Su FD001 escludono `setting_1` e tengono le altre 17. Su FD003
escludono `cycle`, `setting_2`, `sensor_07` e `sensor_12`.

La ricerca esaustiva su 262.143 e 524.287 sottoinsiemi trova quindi esattamente
ciò che trovano le due ricerche direzionali con 18 e 19 valutazioni, in 11,0 e
22,4 secondi contro centesimi di secondo. È un risultato negativo sul valore
della ricerca esaustiva su questi dati: la struttura del problema non presenta
le interazioni fra variabili che rendono subottimali le ricerche greedy.

Su FD003 la selezione esclude il numero di ciclo e ottiene 19,8486, contro i
19,85 ± 1,45 già registrati per la regressione lineare senza numero di ciclo
nella convalida del protocollo. Le due misure, ottenute per vie diverse, si
confermano a vicenda.

ESITO del controllo con selezione annidata. Su FD001 la selezione rifatta dentro
ciascun fold produce 20,35 contro i 20,34 riportati, cioè un ottimismo di 0,01
cicli, con cardinalità selezionate fra 15 e 17. Su FD003 produce 20,12 ± 1,33
contro i 19,85 riportati, cioè un ottimismo di 0,27 cicli, con cardinalità fra
13 e 16.

Su FD003 la selezione delle variabili guadagna 0,086 cicli sulla regressione
lineare multipla, mentre l'ottimismo introdotto dal modo in cui quel guadagno è
misurato vale 0,27 cicli, tre volte tanto. Il primo posto dei metodi di
selezione nella tabella di FD003 è quindi un effetto del protocollo e non una
proprietà dei modelli. La variabilità delle cardinalità selezionate fra fold
conferma che il minimo della curva è instabile perché la curva è piatta.

### Stabilità dei coefficienti

Il bootstrap sui motori individua come coefficienti di segno non stabile
`setting_1` e `setting_2` su FD001, e gli stessi due più `sensor_07` su FD003.
Le variabili escluse dalla selezione esaustiva sono `setting_1` su FD001 e
`cycle`, `setting_2`, `sensor_07`, `sensor_12` su FD003.

Le due procedure non condividono criterio: una misura la stabilità del segno su
ricampionamenti dei motori, l'altra minimizza un errore in cross-validation.
L'indicazione convergente su `setting_1`, `setting_2` e `sensor_07` è quindi
sostenuta da evidenza indipendente.

### Notebook di analisi

`notebooks/02_modelli_lineari.ipynb` legge gli artefatti prodotti dai due script
e ne ricava sei figure in `results/figures/` e otto tabelle in
`results/tables/`. Non addestra modelli e non ricalcola nulla, quindi si esegue
in pochi secondi.

Il testo interpretativo del notebook spiega come si legge ciascuna figura senza
incorporare i valori numerici, che stanno nelle tabelle esportate e vengono
rigenerati a ogni esecuzione. Motivo: un commento con i numeri scritti dentro
diventerebbe falso a ogni riesecuzione degli esperimenti. Le due affermazioni
più forti del blocco, la coincidenza dei sottoinsiemi selezionati dai tre metodi
e la convergenza delle procedure sulle variabili non informative, sono verificate
da codice invece che asserite nel testo.

### Limiti dichiarati

I punteggi di questo blocco sono ottimisticamente distorti, perché la
cross-validation non è annidata e la ricerca degli iperparametri usa le stesse
partizioni su cui la prestazione viene poi misurata. L'entità della distorsione
è stata quantificata per il caso più esposto, i metodi di selezione delle
variabili, ed è risultata trascurabile su FD001 e superiore al vantaggio del
metodo su FD003.

Tutte le misure sono in cross-validation. Nessuna riga dell'insieme di verifica
ufficiale è stata letta in questo blocco.

Il motore di stima usato dai metodi di selezione è codice del progetto e non di
libreria. La sua equivalenza con la stima ordinaria è verificata dal codice, ma
resta un punto in cui il progetto non si appoggia a un'implementazione di
riferimento.

La sezione sulle procedure di ricampionamento opera su 100 unità e non su
20.631 righe, quindi le sue stime hanno la variabilità che compete a un
campione di cento elementi.

## [27-08-2026] — Sequenza dei blocchi residui del confronto

Il perimetro delle tecniche di regressione rimaste fuori dal blocco lineare è stato
ricostruito dal materiale dei laboratori e tradotto in tre blocchi sperimentali più uno
di chiusura.

- Blocco 2, superamento della linearità (laboratorio 8): regressione polinomiale, step
  functions, regression spline, modello additivo generalizzato.
- Blocco 3, famiglia ad albero (laboratori 9 e 10): albero potato per cost-complexity,
  bagging, foresta casuale, AdaBoost, gradient boosting, XGBoost.
- Blocco 4, kernel e reti (laboratorio 11): macchine a vettori di supporto con kernel
  lineare, radiale e polinomiale, percettrone multistrato.
- Blocco 5, chiusura: graduatoria complessiva, lettura unica dell'insieme di verifica
  ufficiale, controllo di sensibilità sulla soglia di censura, report e README.

Motivo dell'unione dei laboratori 9 e 10 in un blocco solo: la lettura centrale di quella
famiglia è albero singolo contro bagging contro foresta casuale contro boosting, cioè
riduzione della varianza per aggregazione contro riduzione della distorsione per
addizione. Distribuirla su due blocchi la trasformerebbe in un rimando fra artefatti
invece che in un confronto dentro una tabella sola.

Alternativa scartata: anticipare il blocco 4 per misurare presto il costo delle macchine a
vettori di supporto, che è il rischio maggiore del piano. Scartata perché per conoscere
quel costo basta misurarlo, non serve riordinare i blocchi, e l'ordine scelto segue la
progressione del programma del corso.

I metodi non supervisionati del laboratorio 12 sono collocati nel blocco di chiusura in
forma circoscritta: raggruppamento delle cento traiettorie di ciascun sottoinsieme,
descritte da poche statistiche per motore, con K-Means e clustering gerarchico, per
verificare se i due modi di guasto di FD003 siano separabili e se l'errore dei modelli si
concentri su un gruppo. Motivo: non sono tecniche di regressione e la consegna non le
richiede, quindi entrano soltanto dove servono al commento.

Il controllo di sensibilità sulla soglia di censura resta collocato nel blocco di
chiusura, dove il modello migliore su cui eseguirlo esiste.

## [27-08-2026] — CORREZIONE: le step functions corrispondono a una classe di libreria

La voce del 20-08-2026 registra che le step functions non corrispondono a una classe di
scikit-learn e vanno costruite come variabili indicatrici su intervalli. L'affermazione è
sbagliata: `KBinsDiscretizer` con codifica a indicatrici dense costruisce esattamente le
colonne che il laboratorio ottiene a mano con `np.digitize`, e in più adatta i punti di
taglio dentro la pipeline, quindi sulla sola parte di addestramento di ciascun fold.

Il modello del blocco 2 usa la classe di libreria. La costruzione a mano avrebbe
duplicato codice già disponibile e testato, senza aggiungere controllo su nulla.

## [27-08-2026] — Blocco del superamento della linearità: griglie, esecuzione e risultati

### Impostazione del blocco

Le quattro tecniche del laboratorio 8 sono applicate a tutte le variabili e non a una per
volta come nella parte didattica del materiale. Motivo: un modello costruito su una sola
variabile non sarebbe confrontabile con quelli degli altri blocchi, che usano l'intera
matrice, e non risponderebbe alla consegna.

Ogni modello è una pipeline in cui la trasformazione precede una regressione lineare,
quindi la trasformazione è adattata dentro ciascun fold come la standardizzazione.

La regressione polinomiale usa l'espansione con interazioni, che è la forma predefinita
dello strumento del laboratorio. Motivo: è l'unico modello del blocco non additivo, e
senza le interazioni il blocco non conterrebbe alcun modello capace di rappresentare
l'effetto congiunto di due variabili. Il grado 1 resta in griglia perché è il caso in cui
l'espansione non aggiunge nulla e il modello coincide con la regressione lineare
multipla.

La collocazione dei punti di taglio delle step functions e dei nodi delle spline
(intervalli di ampiezza uguale, come nel laboratorio, oppure tagli sui quantili) è un
iperparametro invece che una scelta fissata a priori. Motivo: il costo aggiuntivo è di
pochi secondi e la scelta diventa misurata invece che asserita. Il controllo sui bordi non
si applica a un parametro con due soli valori, ed è corretto che non si applichi, perché
su un parametro non ordinato la posizione estrema non ha significato.

### Adattatore per il modello additivo

Sintomo: `pygam.LinearGAM` supera la clonazione ma fa fallire con `AttributeError`
l'adattamento di qualunque `Pipeline` che lo contenga.

Causa: la classe non implementa `__sklearn_tags__`, che scikit-learn richiede dalla
versione 1.6 agli stimatori usati in composizione.

Soluzione: `src/nonlinear.py` contiene `GamRegressor`, adattatore che eredita da
`BaseEstimator`, espone gli iperparametri come argomenti del costruttore e costruisce i
termini dentro `fit`, quando il numero di colonne è noto. Questo lo rende anche
indipendente dal sottoinsieme, che ha 18 variabili su FD001 e 19 su FD003. Verificati
adattamento, predizione, clonazione, ricerca su griglia e lettura delle funzioni parziali.

### Griglie e costi, fissati prima dell'esecuzione

| Modello | Griglia | Configurazioni | Costo per addestramento |
|---|---|---|---|
| Polinomiale | grado in {1, 2, 3} | 3 | da 0,3 a 3,5 s |
| Step functions | intervalli in {3, 5, 8, 12, 20, 30}, tagli in {uguali, quantili} | 12 | da 0,1 a 0,3 s |
| Spline | nodi in {3, 5, 8, 12, 20}, grado in {1, 2, 3}, tagli in {uguali, quantili} | 30 | da 0,2 a 0,4 s |
| Additivo | penalizzazione su 9 valori da 1e-3 a 1e5, funzioni di base in {5, 10, 20} | 27 | da 0,8 a 2,8 s |

I costi sono misurati prima di fissare le griglie su matrici della forma di quelle del
progetto. Il grado 4 del polinomio non è in griglia: genera 7.314 colonne, richiede circa
140 secondi per addestramento e 772 MB per la sola matrice espansa, e vi entrerebbe solo
se la regola sui bordi lo imponesse.

### Applicazione della regola sui bordi

La prima esecuzione ha selezionato configurazioni estreme in quattro punti: numero di
funzioni di base del modello additivo al massimo su entrambi i sottoinsiemi, numero di
intervalli delle step functions al massimo su FD003, grado della spline al minimo su
FD003. Le griglie sono state estese a intervalli fino a 80, funzioni di base fino a 40 e
grado fino a 0, su entrambi i sottoinsiemi e non sul solo sottoinsieme in cui il bordo era
stato toccato. Motivo: griglie diverse sui due sottoinsiemi renderebbero le due repliche
del confronto non più condotte sotto lo stesso protocollo.

Le prime due estensioni hanno risolto il bordo: le configurazioni selezionate sono rimaste
rispettivamente a 30 intervalli e a 20 funzioni di base, ora interne.

### Problema tecnico: configurazioni scomparse senza traccia

Sintomo: la seconda esecuzione ha prodotto centinaia di eccezioni `ValueError` dentro la
trasformazione spline, e ciononostante ha riportato un vincitore regolare e nessun avviso
sui bordi.

Causa immediata: la base spline di grado 0 con estrapolazione costante fallisce su
qualunque valore fuori dall'intervallo osservato in addestramento, condizione che si
verifica in ogni fold. Il difetto è circoscritto a quella combinazione: sono state provate
tutte e dodici le combinazioni di grado (0, 1, 2, 3) ed estrapolazione (costante,
prosecuzione, lineare) e le altre undici funzionano.

Causa radice: quando una configurazione solleva un'eccezione, la ricerca su griglia di
scikit-learn le assegna punteggio non definito e prosegue. La configurazione sparisce
dalla graduatoria senza lasciare traccia, e nulla distingue una griglia valutata per
intero da una in cui una parte non è mai stata provata. La conseguenza sul protocollo è
diretta: il controllo sui bordi diventa privo di significato se la zona verso cui la
griglia è stata estesa è proprio quella che non viene valutata.

Soluzione, su due piani. `src/search.py` conta le configurazioni con punteggio non
definito e le riporta negli artefatti insieme alle mancate convergenze; il contatore è
verificato con uno stimatore che fallisce di proposito su un valore della griglia.
L'errore non viene fatto sollevare, perché una singola configurazione difettosa
interromperebbe un esperimento intero mentre così l'esecuzione resta utilizzabile e il
fatto resta registrato.

L'estensione della griglia del grado è stata ritirata, per due ragioni indipendenti. Il
grado 0 produce funzioni indicatrici su intervalli di ampiezza uguale, che sono le step
functions già presenti in tabella come modello a sé: il limite inferiore porta a un
modello noto e non a uno spazio nuovo, come accade a Elastic Net verso il lato di Ridge.
E quella configurazione non è valutabile. L'estremo selezionato sul grado minimo della
spline su FD003 resta quindi come limite dichiarato.

Alternativa scartata: cambiare l'estrapolazione per rendere valutabile il grado 0.
Cambiarla per la sola configurazione difettosa introdurrebbe una differenza di trattamento
dentro la griglia di uno stesso modello; cambiarla per tutte significherebbe modificare il
modello dopo averne visto il risultato, e l'estrapolazione che prosegue l'andamento
polinomiale produce fuori intervallo valori di base di ampiezza crescente, quindi
predizioni instabili proprio sulle unità che il modello non ha visto.

### Avviso sugli intervalli degeneri

Sintomo: la discretizzazione emette un avviso, una volta per configurazione e per fold,
quando su una variabile a pochi valori distinti alcuni intervalli risultano di ampiezza
nulla e vengono rimossi. Il registro dell'esecuzione ne risultava illeggibile.

L'esito della trasformazione è corretto: la variabile riceve meno colonne, le altre non
sono toccate. L'avviso è filtrato dentro `fit` di una sottoclasse del discretizzatore,
agganciato al testo del messaggio così che un messaggio diverso torni a comparire. In
cambio, lo script registra quante colonne l'espansione ha effettivamente generato contro
quante ne produrrebbe se ogni variabile ricevesse tutti gli intervalli: su FD003 la
configurazione selezionata genera 475 colonne su 570 nominali, su FD001 ne genera 360 su
360. Il fenomeno riguarda solo i tagli sui quantili, perché gli intervalli di ampiezza
uguale hanno larghezza positiva per costruzione, e rende visibile una proprietà del dato
che i modelli lineari non mostravano: alcune letture di sensori sono quantizzate dallo
strumento di misura.

### Costruzione delle baseline spostata nel motore

La costruzione delle due baseline era dentro lo script del blocco lineare. Ogni blocco del
confronto ne ha bisogno per rendere leggibile la propria tabella, quindi è stata spostata
in `src.experiment` e lo script del blocco lineare la importa invece di ridefinirla. Il
comportamento è identico e le baseline riproducono al centesimo i valori già registrati,
il che verifica anche che matrice di progetto e partizioni siano rimaste quelle di agosto.

### Risultati

FD001, media e dispersione su 15 partizioni.

| Modello | Configurazione | Termini | RMSE | MAE | R quadro |
|---|---|---|---|---|---|
| Modello additivo | penalizzazione 100, 20 funzioni di base | 18 | 17,46 ± 1,21 | 13,00 ± 0,91 | 0,823 |
| Regression spline | grado 2, tagli uguali, 5 nodi | 88 | 17,47 ± 1,19 | 13,05 ± 0,90 | 0,823 |
| Step functions | 20 intervalli, tagli uguali | 324 | 17,69 ± 1,20 | 13,19 ± 0,89 | 0,819 |
| Regressione polinomiale | grado 2 | 189 | 17,75 ± 1,28 | 13,81 ± 1,02 | 0,817 |
| Solo numero di ciclo | baseline | 1 | 27,88 ± 2,47 | 21,49 ± 1,71 | 0,549 |
| Predizione costante | baseline | 0 | 41,69 ± 0,14 | 36,98 ± 0,12 | -0,002 |

FD003, media e dispersione su 15 partizioni.

| Modello | Configurazione | Termini | RMSE | MAE | R quadro |
|---|---|---|---|---|---|
| Regression spline | grado 1, tagli uguali, 12 nodi | 203 | 15,91 ± 1,10 | 12,05 ± 0,73 | 0,846 |
| Modello additivo | penalizzazione 10, 20 funzioni di base | 19 | 15,95 ± 1,09 | 12,06 ± 0,74 | 0,845 |
| Step functions | 30 intervalli, tagli sui quantili | 475 | 16,10 ± 1,08 | 12,20 ± 0,77 | 0,842 |
| Regressione polinomiale | grado 2 | 209 | 16,44 ± 1,02 | 12,51 ± 0,84 | 0,835 |
| Solo numero di ciclo | baseline | 1 | 35,12 ± 2,64 | 27,53 ± 2,12 | 0,244 |
| Predizione costante | baseline | 0 | 40,73 ± 0,65 | 35,55 ± 0,14 | -0,007 |

ESITO: il blocco supera il blocco lineare su entrambi i sottoinsiemi. Il miglior modello
lineare valeva 20,33 ± 1,14 su FD001 e 19,85 ± 1,45 su FD003; il migliore di questo blocco
guadagna 2,87 cicli su FD001 e 3,94 su FD003, cioè 2,4 e 3,1 dispersioni. È il primo
divario del progetto che superi la soglia di leggibilità fissata dal protocollo, e
conferma l'ipotesi con cui il blocco lineare si era chiuso: il limite stava nella forma
della relazione fra letture dei sensori e vita residua, non nella varianza della stima.

Il guadagno viene dall'additività non lineare e non dalle interazioni. I tre modelli
additivi stanno davanti alla regressione polinomiale su entrambi i sottoinsiemi, e il
polinomio è l'unico che rappresenta le interazioni.

Il numero di termini non è correlato all'errore: il modello con 18 termini e quello con
475 stanno a una frazione di dispersione l'uno dall'altro. Le curve di validazione
scendono rapidamente e poi restano piatte su un tratto lungo.

Cautele di lettura. I primi due modelli distano 0,00 e 0,03 dispersioni e non vengono
ordinati; che a pareggiare siano spline e modello additivo è coerente con il fatto che
rappresentano la stessa cosa, una funzione liscia per variabile, e differiscono solo per
come ne governano la flessibilità. La configurazione selezionata da ciascuna griglia cade
su un tratto piatto e non va commentata come un ottimo individuato con precisione. I
punteggi restano ottimistici perché la cross-validation non è annidata, e la distorsione
cresce con il numero di configurazioni esplorate, che in questo blocco varia da 3 a 45 fra
i modelli: il confronto fra le righe non è a parità di questo fattore. Il modello additivo
è l'unico del confronto stimato da una libreria diversa da scikit-learn, attraverso un
adattatore scritto per il progetto.

Nessuna riga dell'insieme di verifica ufficiale è stata letta.

### Artefatti

`experiments/nonlinear_models/` contiene per ciascun sottoinsieme tabella di confronto,
metriche per fold, griglie complete, termini con la variabile di provenienza, funzioni
parziali del modello additivo e diagnostica. Il notebook `03_modelli_non_lineari.ipynb`
legge quegli artefatti e produce figure e tabelle finali senza rieseguire lavoro
computazionale.