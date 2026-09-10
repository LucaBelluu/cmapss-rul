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

## [28-08-2026] — Blocco 3: la famiglia ad albero (laboratori 9 e 10)

### Griglie e loro determinazione

I costi sono stati misurati prima di fissare le griglie, con
`scripts/measure_tree_costs.py`, sulla prima partizione del seme di ricerca e
sugli angoli costosi delle griglie candidate. La configurazione piu' onerosa e'
il gradient boosting di scikit-learn a 600 stadi e profondita' 5, con 25,6 s per
adattamento su FD001 e 31,6 su FD003. La stessa configurazione costa mezzo
secondo all'implementazione esterna, con errore che coincide alla seconda cifra.
Un insieme di 300 alberi non potati ha 3,7 milioni di nodi e occupa circa 240 MB,
quantita' che determina il numero di processi utilizzabili nella ricerca e non il
disegno dell'esperimento.

Potatura per cost-complexity. Il laboratorio ricava la sequenza dei valori di
potatura dai dati e sceglie quello che minimizza l'errore sull'insieme di
verifica. Nessuna delle due cose e' trasferibile: la scelta guarderebbe i dati su
cui si misura il risultato, e una sequenza ricavata dai dati cambia da fold a
fold, quindi non definisce una griglia comune ne' alle partizioni ne' ai due
sottoinsiemi. Motivo: la griglia e' un insieme di valori fissato a priori, uguale
ovunque, `[0.0]` seguito da 26 valori logaritmici fra 0,01 e 1000. La scala non e'
arbitraria: il parametro e' nelle unita' dell'impurita', l'impurita' della radice
e' la varianza del target e vale circa 1.700 su entrambi i sottoinsiemi, quindi
l'intervallo copre per costruzione l'intero percorso, dall'albero non potato
all'albero ridotto alla radice.

Alternative scartate: usare la sequenza calcolata sull'intera parte di
addestramento (dipende dal target di tutti i motori, compresi quelli che finiscono
nelle parti di verifica dei fold, e produce griglie diverse sui due sottoinsiemi);
ricalcolarla dentro ogni fold con una selezione interna (tratterebbe l'albero in
modo piu' severo degli altri cinque modelli del blocco, e i quindici fold
sceglierebbero alberi diversi, quindi non esisterebbe un albero potato di cui
mostrare la struttura).

Numero di alberi degli insiemi per aggregazione. Non e' un iperparametro: l'errore
decresce in valore atteso in modo monotono e satura, quindi governa la precisione
di una media e non un compromesso. Metterlo in griglia farebbe selezionare sempre
il valore massimo e chiederebbe alla regola sui bordi un'estensione senza fine.
Motivo: e' fissato a 300, sopra i valori del laboratorio, e la scelta e' verificata
dalla curva di saturazione misurata prima dell'esecuzione. Fra 300 e 500 alberi
l'errore si sposta di 0,010 cicli sul bagging di FD001 e di 0,024 sulla foresta di
FD003, contro una dispersione fra fold di 1,1, e su FD003 non e' monotono: oltre
le poche centinaia di alberi la variazione residua e' rumore della partizione.

Bagging. Con alberi non potati e numero di alberi fissato non ha iperparametri, ed
entra in tabella senza configurazione. E' lo stesso modello della foresta quando
ogni divisione puo' scegliere fra tutte le variabili, e la curva di saturazione lo
conferma: a parita' di numero di alberi i due errori coincidono (16,150 contro
16,150 su FD001, 17,050 contro 17,045 su FD003). E' il controllo di correttezza
analogo alla coincidenza fra regressione sulle componenti principali a componenti
complete e minimi quadrati.

Foresta casuale. La frazione di variabili candidate e' espressa come frazione e non
come conteggio, perche' i due sottoinsiemi hanno 18 e 19 colonne e la griglia deve
restare letteralmente la stessa.

Gradient boosting e implementazione esterna ricevono griglie identiche, sugli
stessi assi e sugli stessi valori. Motivo: il confronto fra le due righe riguarda
cosi' l'implementazione e non il budget di ricerca. La differenza che resta e' la
regolarizzazione esplicita che l'implementazione esterna applica per impostazione
predefinita, che non e' stata azzerata.

### Problema tecnico sulle importanze

Sintomo: la prima esecuzione si interrompe sul bagging con un errore di attributo
mancante. Causa: `BaggingRegressor` e' l'unico insieme di scikit-learn che non
espone `feature_importances_`, mentre foresta, AdaBoost e gradient boosting lo
espongono. Soluzione: l'importanza viene ricostruita come media non pesata delle
importanze dei suoi alberi, che e' la definizione stessa usata dalla foresta,
verificata sulla sorgente (l'importanza restituita da una foresta coincide cifra
per cifra con la media delle importanze dei suoi alberi). La ricostruzione non
introduce quindi una grandezza diversa da quella riportata sulle altre righe.

Seconda differenza fra librerie, individuata prima dell'esecuzione:
`feature_importances_` dell'implementazione esterna restituisce per impostazione
predefinita il guadagno medio per divisione e non lo normalizza, mentre
scikit-learn restituisce la riduzione di impurita' totale pesata e normalizzata a
somma uno. Sotto lo stesso nome la tabella avrebbe contenuto due grandezze diverse
a seconda della riga. Il registro chiede il guadagno complessivo, che e' l'analogo
della grandezza di scikit-learn, e il lettore normalizza tutte le colonne a somma
uno.

L'importanza per permutazione e' calcolata sulla parte di verifica di ciascuna
delle cinque partizioni del seme di ricerca, con il modello riaddestrato sulla
parte di addestramento della stessa partizione. Il laboratorio la calcola
sull'insieme di verifica ufficiale, che qui e' chiuso fino alla chiusura della
graduatoria.

### Catena delle estensioni imposta dalla regola sui bordi

La prima esecuzione completa ha prodotto quattro bordi veri. Le estensioni sono
state applicate su entrambi i sottoinsiemi anche dove il bordo si era manifestato
su uno solo, con un punto per asse e mantenendo la spaziatura propria dell'asse. I
valori di partenza sono rimasti dentro la griglia: toglierli perche' avevano
ottenuto punteggi peggiori sarebbe stata una selezione a posteriori sulla griglia.

Gradient boosting e implementazione esterna avevano selezionato la profondita'
massima su FD003. Estesa a 8, entrambi confermano la configurazione precedente su
entrambi i sottoinsiemi: il bordo non vincolava, e la colonna aggiunta resta in
griglia con i suoi punteggi.

AdaBoost ha richiesto due estensioni e si e' fermato ugualmente sull'angolo della
griglia. L'analisi dell'algoritmo, verificata sulla sorgente di scikit-learn,
spiega perche'. In AdaBoost.R2 il peso di ciascuno stadio e' il tasso di
apprendimento moltiplicato per il logaritmo dell'inverso dell'errore relativo,
quindi il tasso riscala tutti i pesi della stessa costante; l'aggregazione e' una
mediana pesata, che individua lo stadio in cui la somma cumulata dei pesi supera
meta' del totale ed e' percio' invariante a un riscalamento comune. Il tasso non
agisce sulla predizione attraverso i pesi degli stadi, ma soltanto attraverso
l'aggiornamento dei pesi delle osservazioni. Quando tende a zero il ripesaggio si
annulla, ogni stadio viene adattato su un campione bootstrap a pesi uniformi e le
predizioni sono combinate per mediana: quel limite e' il bagging, che il confronto
contiene gia' con una riga propria.

CORREZIONE della classificazione dei bordi fissata all'inizio del blocco: il bordo
inferiore del tasso di apprendimento di AdaBoost era stato elencato fra i bordi
veri, ed e' invece strutturale, nello stesso senso della potatura nulla e della
frazione unitaria di variabili candidate. Sotto non c'e' un modello nuovo, c'e' un
modello gia' in tabella. La correzione non dipende dai punteggi ottenuti e vale
allo stesso modo qualunque fosse stato il risultato. La proprieta' riguarda il
solo asse del tasso: sull'asse della profondita' l'estensione e' rimasta dovuta,
perche' con un tasso interno alla griglia il modello non degenera e alberi di base
piu' profondi sono un modello diverso, non un modello gia' presente.

Criterio di chiusura della catena, fissato prima di lanciare l'ultima estensione:
un'estensione che sposta il modello di meno della dispersione fra fold ha
raggiunto la regione in cui il protocollo non distingue, e continuare
inseguirebbe differenze che la regola di lettura del progetto dichiara
illeggibili. L'ultima estensione ha spostato AdaBoost di 0,24 su FD001 (dispersione
1,27) e di 0,46 su FD003 (dispersione 1,18), quindi sotto la soglia su entrambi, e
la catena e' stata chiusa. Il criterio vale come condizione di arresto di una
catena gia' iniziata, non come motivo per non applicare la regola sui bordi.

### Risultati

FD001, RMSE medio e dispersione sulle 15 partizioni di confronto.

| Modello | Configurazione | RMSE | MAE | R quadro |
|---|---|---|---|---|
| Foresta casuale | frazione 0,33, foglia minima 5 | 16,59 ± 1,46 | 11,58 | 0,840 |
| XGBoost | tasso 0,05, profondita' 3, 300 stadi | 16,70 ± 1,36 | 11,61 | 0,838 |
| Gradient boosting | tasso 0,05, profondita' 3, 300 stadi | 16,71 ± 1,37 | 11,62 | 0,838 |
| Bagging | nessun iperparametro | 16,89 ± 1,44 | 11,62 | 0,834 |
| AdaBoost | tasso 0,01, profondita' 6, 800 stadi | 16,97 ± 1,27 | 12,24 | 0,833 |
| Albero potato | ccp_alpha 0,631, 57 foglie, profondita' 10 | 18,48 ± 1,27 | 12,78 | 0,802 |
| Solo numero di ciclo | baseline | 27,88 ± 2,47 | 21,49 | 0,549 |
| Predizione costante | baseline | 41,69 ± 0,14 | 36,98 | -0,002 |

FD003.

| Modello | Configurazione | RMSE | MAE | R quadro |
|---|---|---|---|---|
| XGBoost | tasso 0,05, profondita' 5, 300 stadi | 14,56 ± 1,11 | 9,35 | 0,871 |
| Gradient boosting | tasso 0,05, profondita' 5, 300 stadi | 14,58 ± 1,14 | 9,36 | 0,870 |
| Foresta casuale | frazione 0,33, foglia minima 5 | 14,70 ± 1,21 | 9,45 | 0,868 |
| Bagging | nessun iperparametro | 14,94 ± 1,11 | 9,44 | 0,864 |
| AdaBoost | tasso 0,01, profondita' 6, 800 stadi | 15,52 ± 1,18 | 10,72 | 0,853 |
| Albero potato | ccp_alpha 0,398, 119 foglie, profondita' 13 | 16,87 ± 1,13 | 10,71 | 0,827 |
| Solo numero di ciclo | baseline | 35,12 ± 2,64 | 27,53 | 0,244 |
| Predizione costante | baseline | 40,73 ± 0,65 | 35,55 | -0,007 |

ESITO: i cinque insiemi non sono ordinabili sotto la regola di lettura, con 0,27
dispersioni fra il primo e l'ultimo su FD001 e 0,84 su FD003. L'albero singolo e'
l'unico modello della famiglia che se ne stacca, indietro di 1,38 e 2,07
dispersioni. La famiglia non produce un vincitore, produce un plateau.

Il confronto con i blocchi precedenti e' asimmetrico fra i due sottoinsiemi. Su
FD003 il miglior modello ad albero sta a 1,2 dispersioni dalle spline (14,56 ± 1,11
contro 15,91 ± 1,10) ed e' un vantaggio leggibile. Su FD001 la distanza e' 0,6
dispersioni (16,59 ± 1,46 contro 17,46 ± 1,21), quindi sotto la risoluzione del
protocollo: su quel sottoinsieme la famiglia ad albero non batte in modo
difendibile il blocco non lineare.

La separazione fra bagging e foresta va nella direzione attesa su entrambi i
sottoinsiemi ma vale 0,21 e 0,34 dispersioni: l'effetto della decorrelazione e'
visibile nel segno e non nella misura del confronto. Si legge invece bene nelle
importanze per permutazione, dove il peso del numero di ciclo scende da 12,41 a
7,87 su FD001 e da 14,82 a 5,70 su FD003 passando dal bagging alla foresta.
Obbligando ogni divisione a scegliere fra un terzo delle variabili, la foresta
costruisce percorsi ridondanti, e mescolare il numero di ciclo lascia intatta
l'informazione che i sensori portano al suo posto.

AdaBoost si ferma a 16,97 e 15,52, appena sopra il bagging, coerentemente con
l'analisi che ne indica il bagging come limite della direzione verso cui la
ricerca si muoveva.

Limiti di questa lettura. Il numero di configurazioni esplorate varia da 1 a 100
fra le righe della stessa tabella: il bagging, che non ha griglia, non paga
distorsione da selezione, mentre AdaBoost ne paga quanto cento configurazioni
valutate sulle stesse partizioni su cui il punteggio viene poi riportato. Il
protocollo e' identico per tutti, ma il confronto non e' a parita' di questo
fattore. AdaBoost e' inoltre riportato su una configurazione d'angolo, quindi al
miglior valore della griglia esplorata e non al suo ottimo: la direzione di
miglioramento punta verso la sua degenerazione in un modello gia' presente nel
confronto. Il costo dell'applicazione integrale della regola sui bordi si e'
concentrato su questo modello, che ha assorbito circa meta' del tempo di calcolo
del blocco per collocarsi ultimo fra i cinque insiemi.

## [28-08-2026] — Notebook di analisi del blocco ad albero

Il notebook `notebooks/04_modelli_ad_albero.ipynb` segue la struttura dei tre
precedenti e non esegue lavoro computazionale: legge gli artefatti prodotti dai
due script del blocco e ne ricava figure e tabelle. Unica eccezione, l'albero
potato, che viene caricato gia' adattato da `{SUBSET}_pruned_tree.joblib` perche'
la sua struttura va disegnata e non descritta. Il file serializzato non e'
versionato, come tutto cio' che sta in `experiments/`; la figura che ne deriva lo
e'.

Quattro sezioni sono proprie di questo blocco e non hanno equivalente nei
notebook precedenti.

Il confronto fra le due implementazioni di gradient boosting appaia le 36
configurazioni della griglia condivisa e le rappresenta contro la bisettrice. E'
la lettura per cui le due griglie erano state fatte identiche: gli scarti sono
dell'ordine del centesimo di ciclo e i tempi di ricerca differiscono di due
ordini di grandezza, quindi la differenza fra le due righe della tabella di
confronto e' computazionale e non statistica. Lo scarto residuo ha causa nota
nella regolarizzazione predefinita dell'implementazione esterna, che non e' stata
azzerata.

La curva di saturazione degli insiemi per aggregazione, prodotta dalla sonda dei
costi, entra nel notebook con due funzioni: giustificare il numero di alberi
fissato invece che cercato, e mostrare la sovrapposizione fra bagging e foresta a
variabili complete, che e' il controllo di correttezza del blocco. La figura
mostra anche che su una singola partizione il segno dell'effetto della
decorrelazione non coincide con quello osservato sulle 15 partizioni di
confronto: e' l'esempio piu' diretto, dentro questo blocco, del motivo per cui il
protocollo non riporta numeri di una partizione sola.

La struttura dell'albero potato e' disegnata sui primi tre livelli, e le soglie
di divisione sono riportate anche nelle unita' originali delle variabili
invertendo lo standardizzatore adattato. Motivo: la standardizzazione e' dentro
la pipeline di tutti i modelli per parita' di condizioni, e senza la conversione
le soglie sarebbero in unita' standardizzate e il disegno non sarebbe leggibile.

Le due misure di importanza sono tenute separate e confrontate: mappa di calore
della riduzione di impurita' sui sei modelli, barre della permutazione in cicli,
e una tabella con il peso del numero di ciclo sotto entrambe, la concentrazione
dell'importanza e la concordanza fra i due ordinamenti.

Nella diagnostica e' inclusa la verifica che il bordo sul numero di stadi di
AdaBoost non fosse apparente. L'implementazione interrompe il boosting se uno
stadio supera la meta' dell'errore relativo, e in quel caso le configurazioni a
400 e a 800 stadi avrebbero punteggi identici a parita' di tasso e profondita'.
Non li hanno: gli stadi sono stati adattati tutti e il bordo era reale.

Il notebook produce nove tabelle in `results/tables/` e sette figure in
`results/figures/`, versionate come prova tracciabile dei risultati.

## [30-08-2026] — Blocco dei metodi a margine e delle reti: SVR con tre kernel e percettrone multistrato

### Sonda dei costi, preliminare alle griglie

La stima a margine con kernel ha costo che cresce fra il quadrato e il cubo del numero di righe,
e le righe di addestramento per fold sono 16.435 su FD001 e 19.303 su FD003: il blocco poteva
risultare impraticabile sotto il protocollo del progetto. Le griglie sono state fissate dopo una
misura e non su una stima, con `scripts/measure_kernel_costs.py`, sulla prima partizione del seme
di ricerca.

Le misure escludono il problema. A dimensione piena, sulla configurazione centrale, l'adattamento
richiede 4,6 s con kernel lineare, 1,8 con radiale e 3,5 con polinomiale su FD001, e 6,9, 2,3 e
5,4 su FD003. Sull'angolo con penalizzazione 100 e banda 1 ciclo si sale a 26,6, 7,7 e 23,7 su
FD001 e a 42,0, 12,1 e 42,1 su FD003. L'esponente empirico misurato fra i due punti piu' grandi
della scala vale fra 1,83 e 2,06 sui sei casi: la crescita e' quadratica, non cubica.

Motivo della decisione conseguente: il blocco gira sotto il protocollo pieno, sulla matrice
intera, senza sottocampionamenti e senza trattamenti differenziati. Sono state valutate e
scartate tre alternative, tutte predisposte per il caso in cui il costo fosse risultato
proibitivo. Il diradamento delle righe di addestramento dentro il fold, scartato perche' non
necessario e perche' avrebbe reso questa famiglia l'unica valutata su una matrice ridotta. La
stima in forma primale del solo kernel lineare, misurata (0,7 s contro 26,6 alla stessa
configurazione) e scartata perche' risolve lo stesso problema con un ottimizzatore diverso, non
compare nel materiale del laboratorio e non serve piu' una volta caduto il vincolo di costo. La
riduzione del protocollo per la sola famiglia a margine, scartata a priori perche' avrebbe rotto
la parita' del protocollo di valutazione, che e' il fondamento del confronto.

La dimensione della cache del kernel resta al valore predefinito: 200, 500 e 1.000 MB producono
tempi identici a un centesimo di secondo. La matrice del kernel a dimensione piena occuperebbe
circa 2,2 GB e non entra in cache a nessuna di quelle dimensioni.

### Scala della banda di insensibilita'

La griglia del laboratorio non e' trasferibile. Il laboratorio usa 0,1 su un target con
deviazione standard circa 1,15, cioe' circa il 9 per cento della dispersione; qui il target e' in
cicli e ha deviazione standard circa 41, quindi lo stesso rapporto vale circa 4 cicli.
Trascrivere il valore alla lettera avrebbe reso vettore di supporto quasi ogni riga, con effetto
simultaneo sulla correttezza della specificazione e sul costo della stima. La griglia adottata
copre da 0,5 a 16 cicli, dove la frazione di vettori di supporto misurata passa dal 96 al 43 per
cento.

### Condizionamento del kernel polinomiale

Sintomo osservato: nella sonda, la configurazione con ampiezza 0,5 su un quinto delle righe ha
richiesto 43,9 milioni di iterazioni e 136,7 s su FD001, 73,9 milioni e 261,8 s su FD003, senza
alcun avviso di mancata convergenza.

Causa radice: il valore del kernel polinomiale e' il prodotto interno fra due righe moltiplicato
per l'ampiezza ed elevato al grado. Con 18 colonne standardizzate il prodotto interno e'
dell'ordine delle 18 unita', quindi oltre l'inverso del numero di colonne la matrice del kernel
assume valori di ampiezza crescente e il problema che l'ottimizzatore risolve diventa mal
condizionato. Non e' mancata convergenza: e' convergenza lentissima verso una soluzione
inaffidabile. La misura a dimensione piena lo conferma: ad ampiezza 0,15 e grado 2 l'errore vale
110,1 cicli su FD001 e 18,5 su FD003, cioe' un fattore sei di differenza fra due sottoinsiemi che
tutti gli altri modelli trattano quasi allo stesso modo. Il kernel radiale non ha questo problema
perche' il suo valore resta fra zero e uno.

Soluzioni adottate, tre e distinte.

L'estremo superiore dell'ampiezza del kernel polinomiale e' fissato all'inverso del numero di
colonne, circa 0,06, ed e' dichiarato bordo strutturale prima dell'esecuzione: oltre non c'e' un
modello migliore, c'e' una stima che non converge. Il kernel radiale conserva una griglia che
arriva a 0,5. L'asimmetria fra le due griglie e' motivata dalla matematica del kernel e non dal
tempo di calcolo.

Il grado del kernel polinomiale resta fissato a 3, come nel laboratorio, e non entra in griglia.
Motivo: sull'angolo peggiore il grado 4 richiede 762 s su FD001 e 1.372 su FD003 per singolo
adattamento e in entrambi i casi produce una stima troncata, contro i 24 e 42 s del grado 3 ad
ampiezza interna alla griglia. Un asse meta' dei cui valori produce stime troncate porterebbe in
graduatoria punteggi non confrontabili fra loro. La scelta e' un limite dichiarato del blocco.

Il numero massimo di iterazioni della stima a margine e' fissato a 20 milioni per tutti e tre i
kernel. Il valore sta sopra il fabbisogno della configurazione legittima piu' esigente osservata,
che ne ha richieste 12,7 milioni a dimensione piena convergendo regolarmente, e sotto quello del
caso patologico. E' lo stesso trattamento gia' applicato ai modelli stimati per discesa
coordinata. Nell'esecuzione del blocco la protezione non e' mai intervenuta.

### Arresto della stima della rete

L'arresto anticipato con partizione interna resta disattivato. Motivo: la partizione che
`MLPRegressor` costruisce e' ottenuta mescolando le righe, quindi collocherebbe cicli adiacenti
dello stesso motore da entrambe le parti, che e' la contaminazione che il vincolo di gruppo del
protocollo esiste per escludere, reintrodotta dentro il modello dopo essere stata esclusa fuori.

Il numero massimo di iterazioni entra invece in griglia, a differenza del numero di alberi degli
insiemi per aggregazione del blocco precedente, che e' fissato. Motivo: le due quantita' si
comportano in modo diverso, e la curva misurata prima dell'esecuzione lo mostra. Sull'architettura
a due strati con passo 1e-3, su FD001, la perdita di addestramento scende da 114,0 a 74,0 fra 100
e 3.000 iterazioni mentre l'errore sulla parte di verifica della partizione sale da 15,86 a 18,87
cicli. Le iterazioni non fanno saturare l'errore: lo fanno risalire, quindi governano un
compromesso e vanno selezionate.

Una prima formulazione di questa motivazione affermava che il criterio di arresto interno della
libreria non fosse operativo su questa scala, sulla base del confronto fra la tolleranza (1e-4) e
l'ordine di grandezza della perdita (centinaia di cicli al quadrato). L'affermazione e' errata ed
e' stata corretta: il criterio interviene sulle architetture piu' strette e non su quelle piu'
larghe. La configurazione selezionata su FD001 ha limite di mille iterazioni e si ferma a 592
quando viene riaddestrata sull'intera parte di addestramento, ed e' il motivo per cui le
configurazioni con limite a mille e a duemila iterazioni hanno punteggi identici. La conclusione
che dipendeva da quella affermazione non cambia, perche' poggia sulla curva misurata.

La penalizzazione sui pesi resta al valore predefinito e fuori griglia: la capacita' e' gia'
governata da due assi cercati, l'architettura e il numero di iterazioni, e un terzo asse che
controlla la stessa quantita' avrebbe triplicato la griglia senza aggiungere una dimensione di
scelta distinta.

### Regola sui bordi ed estensioni applicate

Regola fissata prima dell'esecuzione. Bordi veri, che producono estensione se selezionati:
entrambi gli estremi della penalizzazione, con estensione di una decade; entrambi gli estremi
della banda, con dimezzamento verso il basso e raddoppio verso l'alto; entrambi gli estremi
dell'ampiezza del kernel radiale; l'estremo inferiore dell'ampiezza del kernel polinomiale;
entrambi gli estremi del numero di iterazioni della rete e del passo di apprendimento;
l'architettura piu' capiente. Bordo strutturale, che non produce estensione: il solo estremo
superiore dell'ampiezza del kernel polinomiale.

La prima esecuzione ha selezionato configurazioni su bordi in sette casi. Le estensioni sono
state applicate a entrambi i sottoinsiemi anche quando il bordo si era manifestato su uno solo,
perche' griglie diverse fra i due renderebbero le due repliche non piu' condotte sotto lo stesso
protocollo. Gli assi della penalizzazione e della banda hanno smesso di essere condivisi fra i tre
kernel, perche' i bordi toccati sono opposti: la variante lineare e quella polinomiale hanno
toccato il minimo della penalizzazione, quella radiale il massimo. I valori di partenza restano
in griglia con i loro punteggi anche quando peggiori.

La catena si e' chiusa dopo una sola estensione. Spostamento di ciascun modello fra la prima
esecuzione e quella estesa, contro la dispersione fra fold:

| Modello | Prima | Dopo | Spostamento | Dispersione |
|---|---|---|---|---|
| FD001 kernel lineare | 20,318 | 20,320 | +0,002 | 1,12 |
| FD001 kernel radiale | 16,961 | 16,961 | 0,000 | 1,41 |
| FD001 kernel polinomiale | 23,354 | 23,354 | 0,000 | 0,93 |
| FD001 rete | 16,581 | 16,581 | 0,000 | 1,40 |
| FD003 kernel lineare | 19,944 | 19,826 | -0,118 | 1,52 |
| FD003 kernel radiale | 14,676 | 14,676 | 0,000 | 1,37 |
| FD003 kernel polinomiale | 26,335 | 26,335 | 0,000 | 2,67 |
| FD003 rete | 14,045 | 14,042 | -0,003 | 1,06 |

Ogni estensione ha spostato il modello di almeno un ordine di grandezza meno della dispersione fra
fold su entrambi i sottoinsiemi, che e' la condizione di chiusura fissata a priori. Le due
estensioni verso l'alto hanno peggiorato: penalizzazione 1000 sul kernel radiale da' 16,834 contro
16,789 su FD001 e 15,250 contro 14,778 su FD003, e la banda a 32 cicli peggiora tutti i kernel su
entrambi i sottoinsiemi.

Restano selezionati tre estremi dopo l'estensione. La penalizzazione minima del kernel lineare su
entrambi i sottoinsiemi, su un profilo che varia di 0,047 cicli su quattro decadi: la catena si
chiude per movimento insufficiente e non perche' il bordo sia sparito. L'architettura piu'
capiente della rete su FD003, dove il profilo lungo la capacita' e' ripido in basso (27,18 cicli
con otto unita' nascoste) e piatto in cima. L'ampiezza massima del kernel polinomiale su entrambi,
che e' il bordo strutturale e non produce estensione.

### Risultati del blocco

Errore quadratico medio in cross-validation, media e deviazione standard sulle 15 partizioni di
confronto.

| Modello | FD001 | FD003 |
|---|---|---|
| Percettrone multistrato | 16,58 ± 1,40 | 14,04 ± 1,06 |
| SVR, kernel radiale | 16,96 ± 1,41 | 14,68 ± 1,37 |
| SVR, kernel lineare | 20,32 ± 1,12 | 19,83 ± 1,52 |
| SVR, kernel polinomiale | 23,35 ± 0,93 | 26,34 ± 2,67 |
| Baseline sul solo numero di ciclo | 27,88 ± 2,47 | 35,12 ± 2,64 |
| Predizione costante | 41,69 ± 0,14 | 40,73 ± 0,65 |

Configurazioni selezionate: kernel lineare penalizzazione 0,01 e banda 16 su FD001, banda 8 su
FD003; kernel radiale penalizzazione 100, banda 8, ampiezza 0,015 su FD001, penalizzazione 10,
banda 4, ampiezza 0,05 su FD003; kernel polinomiale penalizzazione 10 e banda 16 su FD001,
penalizzazione 0,1 e banda 16 su FD003, ampiezza 0,06 su entrambi; rete a sedici unita' con passo
1e-3 e mille iterazioni su FD001, a due strati da 128 e 64 unita' con passo 1e-4 e 250 iterazioni
su FD003.

Lettura. La rete e il kernel radiale raggiungono il livello del miglior modello dei blocchi
precedenti (16,59 ± 1,46 e 14,56 ± 1,11 per il migliore della famiglia ad albero) e non lo
superano in modo leggibile: i divari valgono 0,01 e 0,52 cicli contro dispersioni sopra l'unita',
quindi la regola di lettura del progetto non ordina questi risultati. Tre famiglie costruite su
principi diversi convergono sullo stesso livello di errore, il che indica che il limite osservato
dipende dal problema e dalla rappresentazione dei dati piu' che dalla classe di modelli.

Il kernel lineare arriva dove arrivano i modelli lineari: 20,32 contro i 20,33 del blocco lineare
su FD001, 19,83 contro 19,85 su FD003. Due stimatori diversi della stessa classe di funzioni,
sotto perdite diverse, producono lo stesso numero a due decimali, ed e' una verifica indipendente
che la catena dati e il protocollo non introducono differenze spurie fra blocchi.

L'asimmetria fra i due sottoinsiemi si presenta qui in una forma nuova: riguarda la capacita'
richiesta e non solo il livello di errore. La rete migliore su FD001 e' la piu' stretta della
griglia, quella su FD003 la piu' larga, con profili di segno opposto lungo lo stesso asse. Due
modi di guasto invece di uno chiedono una funzione piu' articolata.

La frazione di vettori di supporto dei modelli selezionati sta fra il 44,5 e il 64,5 per cento.
La predizione di questi modelli richiede quindi il calcolo del kernel contro decine di migliaia di
righe, e sono i modelli piu' lenti in predizione dell'intero confronto, mentre alberi e rete
rispondono in tempo costante rispetto alla dimensione dell'insieme di addestramento. La differenza
non compare nella metrica.

Nessuna configurazione non valutabile e nessuna stima interrotta dal tetto sulle iterazioni in
tutto il blocco. La ricerca del kernel radiale su FD003 ha richiesto 10.504 s, quasi interamente
nella colonna della penalizzazione 1000 aggiunta dall'estensione e risultata peggiore: e' il costo
normale di applicare la regola sui bordi invece di decidere a posteriori.

### Limiti dichiarati del blocco

La riga del kernel polinomiale e' un limite superiore delle prestazioni della tecnica e non una
sua misura: la griglia dell'ampiezza e' chiusa dal lato in cui il profilo del modello scende
ancora, per una ragione di condizionamento numerico. Alla prima ampiezza successiva la stima non
e' affidabile.

Il grado del kernel polinomiale e' fissato e non selezionato, per la ragione misurata sopra.

La rete e' stimata con un solo seme di inizializzazione dei pesi. La dispersione riportata sui 15
fold contiene quindi la variabilita' dovuta alla partizione ma non quella dovuta
all'inizializzazione, che per questa classe di modelli non e' trascurabile: la dispersione di
quella riga e' sottostimata rispetto alla variabilita' complessiva della procedura.

Il numero di configurazioni esplorate varia da 35 a 175 fra le righe della stessa tabella, quindi
la distorsione ottimistica dovuta alla cross-validation non annidata non e' uniforme fra le righe.

L'importanza per permutazione del kernel lineare su FD003 assegna al numero di ciclo 0,54 cicli
contro i 4,63 dello stesso modello su FD001: la configurazione selezionata li' e' quella con
penalizzazione minima e banda ampia, cioe' una funzione molto piatta, e su una funzione piatta la
permutazione di una singola variabile sposta poco e il peso si ripartisce fra variabili correlate.
Sul kernel polinomiale il numero di ciclo e' in dodicesima posizione su FD001 e in ultima su
FD003, con aumento di errore negativo.

### Artefatti

`src/registry.py` esteso con `KERNEL_MODELS`, `src/margin.py` per le letture strutturali della
famiglia, `scripts/measure_kernel_costs.py` per la sonda dei costi,
`scripts/run_kernel_models.py` per l'esperimento, `notebooks/05_metodi_a_margine_e_reti.ipynb`
per l'analisi. Otto figure e nove tabelle in `results/`.

ESITO: quarto e ultimo blocco del confronto concluso. L'insieme di verifica ufficiale non e' stato
letto in nessuna fase del blocco.

## [31-08-2026] — Chiusura del confronto: graduatoria complessiva sui quattro blocchi

### Perdita e recupero degli artefatti del blocco lineare su FD001

Sintomo: la prima composizione della graduatoria complessiva mostrava, su FD001, una sola riga
del blocco lineare (la regressione lineare multipla) contro le dieci di FD003.

Causa radice: una riesecuzione parziale dello script del blocco lineare sul solo FD001, limitata
a un sottoinsieme dei modelli, ha riscritto `FD001_comparison.csv` e `FD001_cv_folds.csv` con le
sole righe prodotte in quel passaggio. `FD001_selection_history.csv` e gli altri artefatti non
prodotti da quella esecuzione sono sopravvissuti, quindi l'elenco dei file della cartella
appariva completo e l'incoerenza non era visibile guardando la directory. Gli artefatti non sono
versionati, e una sovrascrittura non lascia quindi traccia nella cronologia: il difetto è stato
intercettato dalla composizione della graduatoria e non da un controllo di integrità dei file.

Soluzione: riesecuzione completa del blocco lineare su FD001, con l'elenco intero dei modelli e
il controllo diagnostico con selezione annidata.

ESITO: i valori rigenerati coincidono con quelli registrati il 26-08 su tutte e otto le righe del
blocco e su tutte le configurazioni selezionate. Ridge alpha=1000 a 20,327 ± 1,137, Elastic Net
alpha=0,08685 e bilanciamento 0,05 a 20,329 ± 1,125, i tre metodi di selezione a k=17 con
20,345 ± 1,183, Lasso alpha=0,02812 a 20,345 ± 1,182, regressione lineare multipla a
20,345 ± 1,183, componenti principali con sei componenti a 20,353 ± 1,173. Il controllo con
selezione annidata restituisce 20,35 contro i 20,34 riportati in tabella, con cardinalità
selezionate nei fold pari a 15, 16 e 17. La riproduzione a cinque giorni di distanza, su una
catena nel frattempo modificata da tre blocchi successivi, è la verifica che la riproducibilità
dichiarata è effettiva e non asserita.

### Verifica di identità delle partizioni fra blocchi

Comporre in un'unica graduatoria quattro tabelle prodotte da esecuzioni diverse presuppone che i
punteggi siano stati calcolati sugli stessi motori e sulle stesse righe. Il presupposto non è
garantito dall'unicità del codice del protocollo: una modifica alla catena dati fra
un'esecuzione e l'altra basterebbe a invalidarlo.

La verifica è possibile perché le due baseline sono ricalcolate in ogni blocco, e ogni blocco
contiene quindi una misura indipendente della stessa quantità sulle stesse partizioni. Il
confronto dei conteggi di righe e di motori di ciascun fold è la firma della partizione;
l'uguaglianza dei punteggi la corrobora. La verifica precede la graduatoria e la blocca in caso
di scostamento, invece di accompagnarla con una nota.

ESITO: scarto nullo su tutti e otto i confronti per sottoinsieme, conteggi identici. I quattro
blocchi sono componibili.

### Graduatoria complessiva

Errore quadratico medio in cross-validation, media e deviazione standard sulle 15 partizioni di
confronto. Le configurazioni selezionate sono registrate nelle voci dei rispettivi blocchi.

| Modello | Blocco | FD001 | FD003 |
|---|---|---|---|
| Percettrone multistrato | Margine e reti | 16,58 ± 1,40 | 14,04 ± 1,06 |
| Foresta casuale | Albero | 16,59 ± 1,46 | 14,70 ± 1,21 |
| XGBoost | Albero | 16,70 ± 1,36 | 14,56 ± 1,11 |
| Gradient boosting | Albero | 16,71 ± 1,37 | 14,58 ± 1,14 |
| Bagging di alberi | Albero | 16,89 ± 1,44 | 14,94 ± 1,11 |
| SVR, kernel radiale | Margine e reti | 16,96 ± 1,41 | 14,68 ± 1,37 |
| AdaBoost | Albero | 16,97 ± 1,27 | 15,52 ± 1,18 |
| Modello additivo generalizzato | Non lineare | 17,46 ± 1,21 | 15,95 ± 1,09 |
| Regression spline | Non lineare | 17,47 ± 1,19 | 15,91 ± 1,10 |
| Step functions | Non lineare | 17,69 ± 1,20 | 16,10 ± 1,08 |
| Regressione polinomiale | Non lineare | 17,75 ± 1,28 | 16,44 ± 1,02 |
| Albero di regressione potato | Albero | 18,48 ± 1,27 | 16,87 ± 1,13 |
| SVR, kernel lineare | Margine e reti | 20,32 ± 1,12 | 19,83 ± 1,52 |
| Ridge | Lineare | 20,33 ± 1,14 | 19,88 ± 1,45 |
| Elastic Net | Lineare | 20,33 ± 1,13 | 19,89 ± 1,45 |
| Best subset, forward, backward | Lineare | 20,34 ± 1,18 | 19,85 ± 1,45 |
| Lasso | Lineare | 20,35 ± 1,18 | 19,92 ± 1,46 |
| Regressione lineare multipla | Lineare | 20,35 ± 1,18 | 19,93 ± 1,46 |
| Componenti principali | Lineare | 20,35 ± 1,17 | 19,93 ± 1,46 |
| SVR, kernel polinomiale | Margine e reti | 23,35 ± 0,93 | 26,34 ± 2,67 |
| Baseline sul solo numero di ciclo | | 27,88 ± 2,47 | 35,12 ± 2,64 |
| Predizione costante | | 41,69 ± 0,14 | 40,73 ± 0,65 |

Lettura. Il percettrone multistrato è primo su entrambi i sottoinsiemi, ma il primo posto ha
significato diverso nei due casi. Su FD001 sette modelli stanno entro 0,39 cicli, cioè entro un
terzo della dispersione minima fra fold della tabella: la graduatoria individua un gruppo di
testa e non un vincitore, e la scelta fra quei sette non è decidibile sotto questo protocollo. Su
FD003 il gruppo di testa è più stretto, con la rete davanti a XGBoost di 0,52 cicli contro
dispersioni di 1,06 e 1,11, e la regola di lettura del progetto continua a non separarli.

La struttura per famiglie è invece netta e si ripete identica sui due sottoinsiemi, in tre
gradini: i modelli lineari attorno a 20 cicli, i modelli non lineari additivi fra 15,9 e 17,8,
gli insiemi di alberi con il kernel radiale e la rete in cima. All'interno del blocco lineare
tutti i modelli, dalla regressione senza penalizzazione alla ricerca esaustiva su 262.143
sottoinsiemi, restano entro 0,03 cicli su FD001 e 0,09 su FD003: la penalizzazione e la selezione
delle variabili non hanno niente da recuperare su questa matrice, ed è la classe di funzioni a
essere il limite.

Cautele di lettura. I punteggi sono ottimisticamente distorti perché la cross-validation non è
annidata, e la distorsione non è uniforme fra le righe: il numero di configurazioni valutate sulle
stesse partizioni su cui il punteggio è poi riportato varia da 1 a 524.287 lungo la tabella. La
riga della ricerca esaustiva è quella più esposta, ed è anche quella per cui il controllo con
selezione annidata ha misurato lo scarto (0,01 cicli su FD001, 0,27 su FD003). La riga della rete
è stimata con un solo seme di inizializzazione dei pesi e la sua dispersione non contiene la
variabilità dovuta all'inizializzazione.

### Confronto appaiato fold per fold

Le 15 partizioni sono le stesse per tutti i modelli e per tutti i blocchi, quindi la differenza
fra due modelli si può calcolare fold per fold anziché confrontando due medie con le rispettive
dispersioni. La difficoltà del fold, che è la componente dominante della dispersione riportata in
tabella, è comune ai due modelli e si elide nella differenza. La media delle differenze coincide
per costruzione con la differenza delle medie: l'informazione aggiuntiva sta nella dispersione
della differenza e nella concordanza del segno.

La lettura è fuori dal materiale del corso ed è dichiarata tale. Resta descrittiva: non viene
prodotta alcuna statistica test, e il rapporto fra media e dispersione della differenza non è
convertibile in un livello di significatività, perché i 15 fold condividono le righe di
addestramento e non sono osservazioni indipendenti. La regola di lettura fissata dal protocollo
resta quella principale e questa non la sostituisce.

Su FD001 il divario fra la rete e la foresta casuale vale 0,014 ± 0,223 cicli, con la rete
peggiore in 11 fold su 15: il segno non è nemmeno stabile, e il primo posto è un pareggio pieno.
Su FD003 il divario fra la rete e XGBoost vale 0,515 ± 0,451 con lo stesso segno in 13 fold su
15. La dispersione della differenza è meno della metà di quella dei singoli punteggi, e la
formulazione corretta è che il divario resta sotto la soglia di leggibilità fissata dal
protocollo ma non è distribuito a caso fra i fold. Le due affermazioni convivono e vanno
riportate entrambe.

### Regola di lettura dell'insieme di verifica ufficiale

Fissata qui, prima che esista il codice che quell'insieme lo legge, e non modificabile dopo.

La graduatoria del progetto è quella in cross-validation riportata sopra. L'insieme di verifica
ufficiale serve a misurare il trasferimento fuori campione e non riordina la graduatoria: ha un
solo punteggio per modello, senza misura di variabilità, e ordinare su di esso significherebbe
ordinare su un numero di cui non si conosce l'incertezza.

Vengono letti tutti i modelli selezionati e le due baseline, non i soli migliori, così che il
confronto fuori campione sia disponibile per l'intera tabella e non per la parte che conviene.

Le tre letture prodotte (tutti i cicli delle traiettorie troncate, solo ultimo ciclo, ultimo ciclo
contro target non censurato) non sono confrontabili fra loro né con l'errore in cross-validation,
perché riguardano popolazioni di cicli diverse.

Va inoltre dichiarato che l'insieme di verifica non è rimasto del tutto intatto fino a questo
punto: nella voce del 26-08, in fase di convalida del protocollo, è stato letto per le due
baseline e per la regressione lineare multipla. Quella lettura riguardava modelli senza
iperparametri e non ha condizionato alcuna scelta, ma l'affermazione "letto una sola volta"
presente nei docstring va intesa nel senso che non entra in nessuna selezione, non nel senso
letterale.

### CORREZIONE: significato della colonna del divario in dispersioni

Il commento alla colonna `divario_in_dispersioni` affermava che la scala combinata fra la
dispersione della riga e quella della riga migliore serve a far seguire alla colonna
l'ordinamento dell'errore. L'affermazione è falsa e i dati della graduatoria la smentiscono: su
FD001 la baseline sul solo numero di ciclo ha errore 27,88 e divario 5,64, mentre il kernel
polinomiale ha errore 23,35 e divario 5,72. La baseline è peggiore e risulta più vicina, perché
la sua dispersione vale 2,47 contro 0,93. La scala combinata attenua la distorsione rispetto
all'uso della sola dispersione di riga, ma non la elimina.

La definizione della colonna non cambia: la quantità è quella giusta, cioè una distanza in unità
di dispersione condivisa fra le due righe confrontate, e modificarla imporrebbe di rigenerare
tutte le tabelle dei quattro blocchi. Cambia il commento: la colonna è una distanza e non un
ordine, l'ordinamento è quello della colonna dell'errore, e fra righe di dispersione molto
diversa la colonna può invertirlo. Il calcolo è stato estratto da `comparison_table` in una
funzione dedicata, `gap_in_dispersions`, perché la graduatoria complessiva usa la stessa
quantità e due definizioni separate potrebbero divergere senza che nulla lo segnali.

### Artefatti

`src/final.py` per la composizione dei blocchi, la verifica delle partizioni e il confronto
appaiato; `scripts/run_final_ranking.py` per la produzione della graduatoria;
`src/experiment.py` esteso con `gap_in_dispersions`. Gli artefatti prodotti stanno in
`experiments/final/` e non sono versionati, come per tutti gli altri esperimenti.

ESITO: graduatoria del confronto chiusa. L'insieme di verifica ufficiale non è stato letto in
questa fase.

## [31-08-2026] — Variabilità dei modelli stocastici al variare del seme dello stimatore

Il primo posto della graduatoria è occupato su entrambi i sottoinsiemi da un modello con una
componente casuale interna, e i modelli che lo seguono entro la soglia di leggibilità sono
anch'essi stocastici. Il protocollo fissa un solo seme di stimatore per modello, uguale per
tutti, quindi la dispersione riportata in graduatoria è quella fra le 15 partizioni e non
contiene la variabilità dovuta al seme. Il limite era già dichiarato per la riga della rete nella
voce del 30-08; con quella riga in testa alla graduatoria complessiva, dichiararlo non basta più.

Motivo della misura: senza di essa non è possibile distinguere un primo posto che è proprietà del
modello da uno che è proprietà dell'estrazione.

Vincolo che la misura rispetta: il risultato è diagnostico e non entra in graduatoria, come il
controllo con selezione annidata del blocco lineare. Cambiare la regola sui semi per i soli
modelli stocastici romperebbe la parità del protocollo, che è il fondamento del confronto.

Procedura: la configurazione selezionata di percettrone multistrato, foresta casuale e XGBoost
viene rivalutata sulle stesse 15 partizioni con cinque semi di stimatore. Il primo seme è quello
del protocollo e serve da controllo di fedeltà della ricostruzione: su tutte e sei le righe
riproduce il punteggio in graduatoria con scarto nullo.

Media dell'errore sulle 15 partizioni, dispersione fra i cinque semi ed escursione fra il
migliore e il peggiore.

| Sottoinsieme | Modello | Seme del protocollo | Media fra semi | Dispersione fra semi | Escursione |
|---|---|---|---|---|---|
| FD001 | Percettrone multistrato | 16,581 | 16,654 | 0,044 | 0,110 |
| FD001 | Foresta casuale | 16,595 | 16,599 | 0,002 | 0,006 |
| FD001 | XGBoost | 16,701 | 16,701 | 0,000 | 0,000 |
| FD003 | Percettrone multistrato | 14,042 | 14,038 | 0,026 | 0,062 |
| FD003 | XGBoost | 14,558 | 14,558 | 0,000 | 0,000 |
| FD003 | Foresta casuale | 14,701 | 14,706 | 0,003 | 0,010 |

Lettura, e i due sottoinsiemi danno esiti opposti.

Su FD001 il primo posto non è determinato. Il divario fra rete e foresta casuale vale 0,014
cicli, la dispersione fra semi della rete vale 0,044: il divario sta a un terzo del rumore di
inizializzazione. Il seme del protocollo è inoltre quello che produce il valore migliore dei
cinque, e sulla media fra semi l'ordine si inverte, con la rete a 16,654 contro i 16,599 della
foresta casuale. Il seme è fissato dal protocollo per tutti i modelli e non è stato scelto
guardando i risultati, quindi la graduatoria non è viziata; ma la conclusione difendibile su
FD001 è che rete e foresta casuale non siano ordinabili, e che il primo posto della rete in
tabella sia un effetto dell'estrazione.

Su FD003 il primo posto regge. Il divario dalla rete a XGBoost vale 0,515 cicli contro una
dispersione fra semi di 0,026, cioè venti volte tanto, e il seme del protocollo produce il valore
mediano dei cinque e non il migliore. Il divario resta sotto la soglia di leggibilità fissata dal
protocollo, che si misura sulla dispersione fra fold, ma non è attribuibile all'inizializzazione.

La foresta casuale ha dispersione fra semi di uno o due ordini di grandezza inferiore a quella
della rete. La media su 300 alberi assorbe la variabilità del campionamento delle righe e delle
colonne, mentre la rete ha una sola stima e la sua inizializzazione entra per intero nel
risultato.

XGBoost ha dispersione esattamente nulla su entrambi i sottoinsiemi. Non è stabilità: la
configurazione del registro lascia il campionamento delle righe e delle colonne ai valori
predefiniti, che valgono uno, quindi non c'è alcuna sorgente di casualità e il seme è inerte. La
riga di XGBoost nella tabella sopra non misura una variabilità bassa, misura l'assenza di
variabilità da misurare, ed è un limite del diagnostico su quella riga e non un suo risultato.

Osservazione sul criterio di arresto della rete, contata sugli avvisi emessi
nell'esecuzione. Su FD003 il tetto di 250 iterazioni è raggiunto in tutti e 15 i fold per tutti e
cinque i semi: su quel sottoinsieme il numero di iterazioni è la penalizzazione effettiva del
modello e vincola sempre. Su FD001 il tetto di 1000 iterazioni è raggiunto in 3 casi su 75, tutti
sui semi 1 e 2, e mai con il seme del protocollo: là il criterio di arresto interno interviene
prima, come già registrato il 30-08 per la configurazione selezionata, che si ferma a 592
iterazioni. Il comportamento dello stesso iperparametro è quindi diverso fra i due sottoinsiemi.

### Artefatti

`src/selected.py` ricostruisce i modelli selezionati dagli artefatti dei blocchi, rigenerando la
griglia dal registro e cercando la combinazione la cui etichetta coincide con quella registrata
nella tabella di confronto. La ricostruzione non legge i parametri dalle tabelle: i valori
dell'etichetta sono arrotondati a quattro cifre significative e produrrebbero un modello diverso
da quello valutato. La corrispondenza deve essere unica, e questo rende la ricostruzione anche un
controllo di coerenza fra il registro nella repository e gli artefatti su disco. Risolve inoltre
il caso di parità osservato su FD001, dove due combinazioni della griglia della rete hanno lo
stesso punteggio fino alla quattordicesima cifra e la riga di rango primo nella tabella della
griglia non identifica quella valutata.

`scripts/run_seed_diagnostic.py` esegue la misura e deposita i risultati in `experiments/final/`.

ESITO: la variabilità che il protocollo non cattura è misurata. Su FD001 supera di tre volte il
divario fra i primi due modelli; su FD003 è venti volte più piccola del divario fra i primi due.
L'insieme di verifica ufficiale non è stato letto in questa fase.

## [09-09-2026] — Lettura dell'insieme di verifica ufficiale

I ventidue modelli selezionati e le due baseline sono stati riaddestrati
sull'intera parte di addestramento di ciascun sottoinsieme e valutati una sola
volta sull'insieme di verifica ufficiale, secondo la regola di lettura fissata
prima che esistesse il codice che quell'insieme lo legge.

### Controllo di fedeltà della catena

Le due baseline e la regressione lineare multipla erano già state lette
sull'insieme di verifica in fase di convalida del protocollo, il 26-08. Sono
modelli deterministici e senza iperparametri, quindi i loro punteggi devono
riprodursi: uno scostamento significherebbe che la catena dati è cambiata nel
frattempo e nessuna delle altre righe sarebbe interpretabile. Il controllo è
cablato nello script, precede la lettura dei risultati e blocca l'esecuzione in
caso di scostamento, dopo aver scritto gli artefatti.

ESITO: dodici valori confrontati su due sottoinsiemi, scarto massimo 0,005
cicli, che è l'arrotondamento al centesimo con cui i valori erano stati
registrati. La catena è la stessa a quattordici giorni di distanza.

Nessun avviso di mancata convergenza sulle tre varianti di kernel della macchina
a vettori di supporto, quindi nessuna stima troncata dal tetto alle iterazioni e
nessun punteggio non confrontabile. L'unico avviso proviene dalla rete su FD003,
che raggiunge il tetto di 250 iterazioni: è il comportamento previsto e già
registrato il 30-08, non un difetto della stima.

### Risultati

Radice dell'errore quadratico medio. La colonna della cross-validation è la
media sulle 15 partizioni; le tre letture della verifica sono valori singoli,
senza misura di variabilità. Le quattro colonne riguardano popolazioni di cicli
diverse e i loro valori non si sottraggono fra loro.

| Modello | FD001 CV | FD001 tutti | FD001 ultimo | FD003 CV | FD003 tutti | FD003 ultimo |
|---|---|---|---|---|---|---|
| Percettrone multistrato | 16,58 | 15,52 | 16,72 | 14,04 | 12,65 | 16,01 |
| Foresta casuale | 16,59 | 15,64 | 16,81 | 14,70 | 13,17 | 17,36 |
| XGBoost | 16,70 | 15,66 | 16,58 | 14,56 | 12,92 | 16,26 |
| Gradient boosting | 16,71 | 15,65 | 16,52 | 14,58 | 12,90 | 16,11 |
| Bagging di alberi | 16,89 | 15,87 | 16,95 | 14,94 | 13,41 | 17,23 |
| SVR, kernel radiale | 16,96 | 15,71 | 17,12 | 14,68 | 12,78 | 17,06 |
| AdaBoost | 16,97 | 16,55 | 17,18 | 15,52 | 14,38 | 17,71 |
| Modello additivo generalizzato | 17,46 | 16,41 | 17,15 | 15,95 | 14,54 | 17,56 |
| Regression spline | 17,47 | 16,51 | 17,51 | 15,91 | 14,57 | 17,61 |
| Step functions | 17,69 | 16,56 | 17,23 | 16,10 | 14,66 | 17,90 |
| Regressione polinomiale | 17,75 | 17,46 | 18,48 | 16,44 | 14,24 | 17,74 |
| Albero di regressione potato | 18,48 | 16,97 | 18,67 | 16,87 | 14,33 | 18,06 |
| SVR, kernel lineare | 20,32 | 19,19 | 21,24 | 19,83 | 18,38 | 21,86 |
| Ridge | 20,33 | 19,11 | 21,35 | 19,88 | 18,06 | 21,66 |
| Elastic Net | 20,33 | 19,15 | 21,31 | 19,89 | 18,12 | 21,78 |
| Best subset, forward, backward | 20,34 | 19,07 | 21,43 | 19,85 | 17,97 | 21,27 |
| Lasso | 20,35 | 19,07 | 21,45 | 19,92 | 18,00 | 21,55 |
| Regressione lineare multipla | 20,35 | 19,07 | 21,45 | 19,93 | 17,96 | 21,44 |
| Componenti principali | 20,35 | 19,13 | 21,48 | 19,93 | 17,96 | 21,44 |
| SVR, kernel polinomiale | 23,35 | 20,26 | 23,27 | 26,34 | 19,38 | 28,81 |
| Baseline sul solo numero di ciclo | 27,88 | 23,69 | 32,25 | 35,12 | 26,03 | 36,80 |
| Predizione costante | 41,69 | 35,34 | 41,94 | 40,73 | 31,37 | 43,70 |

La terza lettura, sull'ultimo ciclo contro target non censurato, è negli
artefatti e non in tabella: differisce dalla seconda di circa 1,2 cicli su ogni
riga, quantità coerente con il numero ridotto di unità di verifica la cui vita
residua supera la soglia, e non cambia alcun ordine.

### Concordanza della graduatoria

La correlazione di rango fra la graduatoria in cross-validation e le letture
della verifica, calcolata sui ventidue modelli ed esclusa la coppia di baseline
che è ultima in ogni lettura e gonfierebbe la misura, vale 0,89 su tutti i cicli
e 0,98 sull'ultimo ciclo per FD001, 0,87 e 0,93 per FD003.

Presi da soli quei numeri sono fuorvianti, perché la correlazione di rango
impone un ordine anche fra righe che il protocollo dichiara non ordinabili. La
misura corretta è il conteggio delle inversioni fra le sole coppie che la regola
di lettura del progetto separa, cioè quelle il cui divario in cross-validation
supera la dispersione combinata delle due righe. Sono 136 coppie su 231 in
FD001 e 158 su 231 in FD003.

| Lettura | FD001 | FD003 |
|---|---|---|
| Tutti i cicli | 0 inversioni su 136 | 1 su 158 |
| Solo ultimo ciclo | 0 su 136 | 0 su 158 |
| Ultimo ciclo non censurato | 0 su 136 | 1 su 158 |

Le due inversioni valgono 0,05 cicli (AdaBoost e albero potato) e 0,01 cicli
(foresta casuale e modello additivo). L'ordine che il protocollo dichiara
leggibile si trasferisce quindi integralmente su una popolazione indipendente, e
la correlazione di rango inferiore all'unità è prodotta da riordinamenti interni
a gruppi già dichiarati indistinguibili.

### Il vertice della graduatoria

Su FD001 il primo posto cambia titolare a seconda della lettura: la rete sulla
lettura estesa (15,52), il gradient boosting sull'ultimo ciclo (16,52), con
XGBoost a 16,58, la rete a 16,72 e la foresta casuale a 16,81. Quattro modelli
in 0,3 cicli. Il risultato coincide con quello del diagnostico sul seme dello
stimatore, che aveva mostrato il primo posto della rete attribuibile
all'estrazione: due misure indipendenti, la variabilità fra semi e il
trasferimento fuori campione, portano alla stessa conclusione. Su FD001 il primo
posto non è assegnabile.

Su FD003 la rete è prima in tutte e tre le letture della verifica, come lo era
in cross-validation e come il diagnostico sul seme aveva mostrato non
attribuibile all'inizializzazione. Anche qui le due misure concordano, nella
direzione opposta.

La struttura per famiglie è ciò che regge senza riserve: i tre gradini si
ripetono identici sui due sottoinsiemi e in tutte le letture, e i divari fra
gradini sono di ordine di grandezza superiore alle distinzioni interne.

### Comportamenti che il trasferimento fa emergere

La macchina a vettori di supporto con kernel lineare è la migliore del gradino
lineare in cross-validation su entrambi i sottoinsiemi ed è la peggiore del
gradino in tutte e sei le letture della verifica. Il segno è costante su sei
misure, quindi non è casuale, ma le ampiezze restano sotto la soglia di
leggibilità e il progetto non ordina quelle righe. La configurazione selezionata
ha la penalizzazione minima della griglia e banda di insensibilità di 16 cicli
su FD001 e 8 su FD003: è un modello che non penalizza gli errori sotto quella
soglia, e l'insieme di verifica ha una quota di righe al valore di censura molto
più alta di quella dell'addestramento. Un legame fra le due cose è plausibile ma
non è stato misurato, e resta un'ipotesi.

Su FD003, e solo nella lettura estesa, la regressione polinomiale e l'albero
potato guadagnano quattro posizioni ciascuno e superano i modelli additivi.
L'effetto scompare sull'ultimo ciclo, quindi riguarda la parte iniziale delle
traiettorie, dove il target è appiattito sulla soglia, e non la fase di degrado.

### Cautele di lettura

L'errore sulla verifica è più basso di quello in cross-validation su ogni riga
della tabella. Non è un trasferimento migliore: è la composizione della
popolazione, già misurata il 26-08 sulla regressione lineare. Le traiettorie di
verifica sono troncate in un punto casuale e contengono in proporzione molte più
righe della fase iniziale di vita, dove il target è appiattito sulla soglia; la
deviazione standard del target scende da 41,67 a 27,58 su FD001 e da 40,63 a
24,84 su FD003. Il coefficiente di determinazione si muove nella direzione
attesa e scende.

Ogni lettura della verifica è un valore singolo, privo di misura di variabilità.
La graduatoria del progetto resta quella in cross-validation e non viene
riordinata: le colonne di rango negli artefatti affiancano le posizioni senza
cambiare l'ordine della tabella.

Il conteggio delle inversioni fra coppie separate è una lettura fuori dal
materiale del corso ed è descrittiva: non produce alcuna statistica test.

### Artefatti

`scripts/run_holdout.py`. Produce in `experiments/final/`, per ciascun
sottoinsieme, la tabella delle tre letture, le predizioni su ogni riga di
verifica in forma lunga e la concordanza di rango. Le esecuzioni parziali non
scrivono su disco, per evitare la sovrascrittura di una tabella completa con un
sottoinsieme di righe già osservata sugli artefatti del blocco lineare.

ESITO: l'insieme di verifica ufficiale è stato letto una volta su tutti i
modelli selezionati e sulle due baseline. L'ordine che il protocollo dichiara
leggibile si trasferisce integralmente fuori campione su entrambi i
sottoinsiemi.

## [09-09-2026] — Controllo di sensibilità alla soglia di censura

La censura del target a 125 cicli è un'ipotesi di modellazione fissata a priori e
i valori assoluti di tutte le metriche dipendono da essa. Il controllo misura se
ne dipenda anche l'ordine fra le famiglie di modelli.

### Impostazione

Un modello per famiglia, rivalutato sulle stesse 15 partizioni con la censura
disattivata: Ridge per i modelli lineari, il modello additivo generalizzato per i
modelli non lineari additivi, la foresta casuale per la famiglia ad albero, il
percettrone multistrato per i metodi a margine e le reti. Sono, in ciascuna
famiglia, la riga meglio piazzata su FD001, e sono gli stessi sui due
sottoinsiemi perché le due repliche restino confrontabili.

Il perimetro è più ampio di quello annunciato quando la soglia è stata fissata,
che prevedeva il solo modello migliore e la baseline lineare regolarizzata. Un
modello per famiglia costa quattro rivalutazioni invece di due e permette di
verificare la struttura per gradini, che è il risultato che il confronto
consegna, invece del solo primo posto.

Le due baseline entrano nel controllo insieme ai modelli. Sotto censura
disattivata la predizione costante restituisce la deviazione standard del target
non censurato, che è la scala su cui vanno letti gli errori di quel regime.

La configurazione degli iperparametri resta quella selezionata sotto censura e la
ricerca non viene rifatta. Motivo: rifarla equivarrebbe a condurre un secondo
confronto completo su una diversa definizione del target, che è stato scartato
per costo quando la definizione è stata fissata.

Limite che ne consegue, e che rende la lettura a senso unico: ogni configurazione
è stata scelta per un target di scala diversa da quello su cui viene qui
valutata, quindi una famiglia il cui ottimo si sposta molto risulta
svantaggiata. Se l'ordine regge nonostante questo il risultato è solido; se si
invertisse, non se ne potrebbe concludere che la famiglia è peggiore sotto target
non censurato.

Il regime censurato ripete una misura già in graduatoria e la riproduce con
scarto nullo su tutte e dodici le righe (il massimo osservato è 1,78·10⁻¹⁵ sul
modello additivo di FD003, cioè precisione macchina). La differenza fra i due
regimi è quindi attribuibile alla sola definizione del target.

### Risultati

Il target non censurato ha una scala diversa da quello censurato: su FD001 arriva
a 361 cicli con dispersione 68,88 contro 41,67, su FD003 a 524 cicli con
dispersione 98,85 contro 40,63. Gli errori dei due regimi non sono quindi
confrontabili in valore assoluto e la sola quantità trasferibile è l'ordine
dentro ciascun regime.

FD001, radice dell'errore quadratico medio sulle 15 partizioni.

| Modello | Censurato | Non censurato |
|---|---|---|
| Percettrone multistrato | 16,58 ± 1,40 | 37,14 ± 5,68 |
| Foresta casuale | 16,59 ± 1,46 | 37,20 ± 6,21 |
| Modello additivo generalizzato | 17,46 ± 1,21 | 37,18 ± 5,84 |
| Ridge | 20,33 ± 1,14 | 40,80 ± 5,52 |
| Baseline sul solo numero di ciclo | 27,88 ± 2,47 | 46,50 ± 7,56 |
| Predizione costante | 41,69 ± 0,14 | 68,73 ± 4,31 |

FD003.

| Modello | Censurato | Non censurato |
|---|---|---|
| Percettrone multistrato | 14,04 ± 1,06 | 56,56 ± 10,26 |
| Foresta casuale | 14,70 ± 1,22 | 58,08 ± 10,48 |
| Modello additivo generalizzato | 15,95 ± 1,09 | 60,04 ± 12,14 |
| Ridge | 19,88 ± 1,45 | 62,03 ± 11,46 |
| Baseline sul solo numero di ciclo | 35,12 ± 2,64 | 85,80 ± 17,82 |
| Predizione costante | 40,73 ± 0,65 | 97,32 ± 14,94 |

### Lettura

L'ordine regge. Su FD003 le quattro righe sono nello stesso ordine nei due
regimi. Su FD001 foresta casuale e modello additivo si scambiano, ma nel regime
censurato distano 0,87 cicli contro una dispersione combinata di 1,34, cioè sono
due righe che il protocollo già non ordina: lo scambio non inverte un ordine
leggibile. Ridge resta ultima fra i modelli in entrambi i regimi e su entrambi i
sottoinsiemi, quindi la separazione fra il gradino lineare e gli altri non
dipende dalla soglia.

Il risultato più informativo non riguarda però l'ordine ma la dispersione.
Togliendo la censura la dispersione fra fold passa da 1,1-1,5 a 5,5-6,2 cicli su
FD001 e da 1,0-1,4 a 10,3-12,1 su FD003, mentre il divario fra Ridge e la rete
resta quasi invariato in cicli: 3,75 contro 3,66 su FD001, 5,84 contro 5,47 su
FD003. Misurato nell'unità del protocollo, cioè in dispersioni combinate, quel
divario passa da 2,94 a 0,65 su FD001 e da 4,61 a 0,50 su FD003.

Nel regime non censurato la regola di lettura del progetto non separa quindi più
nessuna coppia fra i quattro modelli, compresa quella che divide la famiglia
lineare dalle altre. La censura non sposta i modelli: cambia la risoluzione con
cui il confronto li distingue. È un argomento a favore della soglia indipendente
da quello con cui era stata fissata, e non è circolare: abbassare la soglia
riduce l'errore per costruzione, ma nulla nella costruzione impone che aumenti
anche il rapporto fra i divari e la dispersione fra fold.

L'incremento di errore prodotto dalla rimozione della soglia è comune alle righe:
da 19,7 a 20,6 cicli su FD001 e da 42,1 a 44,1 su FD003, sulle quattro famiglie e
sulle due baseline. È la misura diretta dell'affermazione con cui la censura era
stata motivata, cioè che la parte di target rimossa dalla soglia contiene una
componente che nessuna classe di funzioni riduce.

Il tetto di iterazioni della rete su FD003 vincola in 15 stime su 15 in entrambi i
regimi, quindi non è un effetto del target non censurato: era già la regola di
arresto operativa sotto censura, come registrato il 31-08. Su FD001 non compare
alcun avviso in nessuno dei due regimi.

### Limiti

Il controllo riguarda quattro modelli su ventidue e una sola soglia alternativa,
che è l'assenza di soglia. Non dice nulla sul comportamento a soglie intermedie.

Le configurazioni sono congelate, quindi il regime non censurato è valutato con
iperparametri scelti per un target di scala diversa.

Nel regime non censurato nessuna coppia è separabile sotto la regola di lettura
del progetto: l'affermazione difendibile riguarda la concordanza dell'ordine e il
segno dei divari, non la loro leggibilità.

### Artefatti

`scripts/run_censoring_sensitivity.py`. Produce in `experiments/final/`, per
ciascun sottoinsieme, le metriche per partizione nei due regimi e la tabella con
media, dispersione e posizione dentro ciascun regime. L'insieme di verifica
ufficiale non viene letto.

ESITO: l'ordine fra famiglie non dipende dalla soglia di censura. La soglia
determina la risoluzione del confronto, non la sua conclusione.

## [09-09-2026] — Raggruppamento delle traiettorie con i metodi non supervisionati

FD001 e FD003 differiscono per il solo numero di modi di guasto, uno contro due,
e il confronto fra modelli mostra su FD003 traiettorie più lunghe, dispersione
delle durate quasi doppia e punteggi migliori. Ho applicato i metodi del
laboratorio 12 per stabilire se quella differenza sia visibile nella forma delle
traiettorie guardandole senza il target, cioè se un metodo non supervisionato
trovi su FD003 una struttura di gruppi che su FD001 non esiste.

Il risultato è strumento di commento e non è una riga del confronto: il task del
progetto è di regressione e i metodi non supervisionati rientrano come strumenti
di esplorazione.

### Impostazione

L'unità di osservazione è il motore e non il ciclo, perché la domanda riguarda la
forma della traiettoria nel suo complesso. Sono usate le sole traiettorie di
addestramento: quelle di verifica sono troncate in un punto casuale, quindi le
statistiche di fine vita non vi sono definite e la durata osservata non è la
durata del motore.

Variabili per motore: per ciascun sensore non costante la lettura media negli
ultimi 10 cicli e la deriva totale (differenza fra la media degli ultimi 10 cicli
e quella dei primi 10), più la durata della traiettoria. Sono 31 variabili su
FD001 e 33 su FD003. Le finestre non si sovrappongono su nessun motore, perché la
traiettoria più breve dura 128 cicli.

Le impostazioni operative sono escluse. Motivo: su questi due sottoinsiemi il
regime di volo è unico e la loro variazione residua è oscillazione di misura
attorno a un valore fisso; la standardizzazione, che il calcolo delle distanze
richiede, la porterebbe a scala piena e ne farebbe rumore dentro la distanza, su
un insieme di cento punti. Nel confronto fra modelli le due variabili sono invece
mantenute, e la differenza di trattamento è ammissibile perché il raggruppamento
non è una riga del confronto.

Le etichette di modo di guasto dei singoli motori non sono distribuite con il
dataset. Dentro un sottoinsieme non esiste quindi un riferimento contro cui
misurare la correttezza di un raggruppamento, e l'indice di Rand corretto è usato
come accordo fra due raggruppamenti diversi. Il numero di gruppi si legge sulla
silhouette.

### Risultati

Silhouette al variare del numero di gruppi, K-Means e gerarchico con
aggregazione di Ward.

| Gruppi | FD001 K-Means | FD001 Ward | FD003 K-Means | FD003 Ward |
|---|---|---|---|---|
| 2 | 0,260 | 0,240 | 0,533 | 0,533 |
| 3 | 0,241 | 0,198 | 0,479 | 0,477 |
| 4 | 0,232 | 0,206 | 0,483 | 0,482 |
| 5 | 0,203 | 0,204 | 0,406 | 0,397 |
| 6 | 0,169 | 0,155 | 0,249 | 0,222 |

Su FD001 non c'è struttura di gruppi. La silhouette massima vale 0,260 e
decresce da lì, e i due algoritmi allo stesso numero di gruppi concordano solo
per 0,702, cioè trovano partizioni diverse.

Su FD003 la struttura c'è ed è a due gruppi. La silhouette vale 0,533 con un
massimo interno, K-Means e Ward producono la stessa identica partizione (accordo
1,000), le dimensioni sono 44 e 56, e la partizione è invariante su tutti e
cinque i semi di inizializzazione. Due algoritmi con funzioni obiettivo diverse
convergono sullo stesso risultato. A parità di costruzione delle variabili e di
numero di unità, 0,533 contro 0,260 è un fattore due.

I criteri di aggregazione complete e average non sono confrontabili con gli altri
su questi dati: a due gruppi isolano un solo motore contro 99 su FD003 e tre
contro 97 su FD001. La loro silhouette è alta perché premia la separazione di
punti isolati, non perché individui una partizione. La colonna della dimensione
minima negli artefatti è ciò che rende leggibile la degenerazione.

Il numero di gruppi trovato su FD003 coincide con il numero di modi di guasto
documentato per quel sottoinsieme. È una coincidenza fra una misura e una
proprietà nota, non un'identificazione: senza etichette per unità, che i due
gruppi siano i due modi resta interpretazione. Ciò che è misurato è che FD003 si
separa e FD001 no.

### Le due popolazioni di FD003

Sui duecento motori dei due sottoinsiemi uniti, dove l'appartenenza al
sottoinsieme è un'etichetta esterna vera, il raggruppamento a due gruppi ottiene
silhouette 0,552 ma accordo di sole 0,191 con quell'etichetta, e produce gruppi
di 44 e 156 unità.

La partizione dominante non è quindi FD001 contro FD003. La tabella incrociata lo
mostra senza ambiguità: i 44 dell'unione sono esattamente i 44 che FD003 separa
da solo, con corrispondenza diagonale e nessuno scambio, e i 100 motori di FD001
finiscono tutti nel gruppo maggiore insieme ai restanti 56 di FD003.

FD003 è composto da una popolazione indistinguibile da FD001 più una seconda
popolazione separata. È il motivo per cui un metodo non supervisionato non
ricostruisce da quale file provenga un motore: per tre quarti delle unità le due
popolazioni si sovrappongono davvero.

Le durate confermano la lettura su una quantità che non è una lettura di sensore.

| Insieme | Motori | Durata media | Dispersione |
|---|---|---|---|
| FD001 | 100 | 206,3 | 46,3 |
| FD003, gruppo che si sovrappone a FD001 | 56 | 202,1 | 42,3 |
| FD003, gruppo separato | 44 | 304,7 | 94,3 |

Ricomponendo i due gruppi di FD003 si ottiene una dispersione complessiva di 86,5
cicli, che è il valore misurato in fase di esplorazione il 26-08. La dispersione
anomala delle durate di FD003, che allora era un fatto isolato, è interamente
prodotta dalla convivenza di due popolazioni, e le due misure si spiegano a
vicenda.

La durata è una delle variabili del raggruppamento, quindi leggere la separazione
attraverso le durate non è di per sé una conferma indipendente. Ho ripetuto il
raggruppamento escludendo la durata: su FD003 la partizione ottenuta coincide con
quella completa (accordo 1,000, gruppi di 44 e 56) e la silhouette sale da 0,533 a
0,542. La separazione è quindi prodotta dalle sole letture dei sensori e la
differenza di durata ne è una conseguenza, non la causa. Su FD001 l'assenza di
struttura resta invariata, con silhouette 0,266.

### Limiti

Le variabili per motore riassumono la traiettoria con lo stato di fine vita e la
deriva complessiva: una traiettoria che degrada in modo non monotono e una che
degrada linearmente fino allo stesso punto sono indistinguibili in questa
rappresentazione.

La silhouette di 0,533 indica una separazione leggibile ma non netta: i gruppi
esistono, non sono isolati.

Il raggruppamento non spiega perché i modelli ottengano su FD003 punteggi
migliori che su FD001. Descrive la struttura della popolazione, non la
difficoltà del problema di regressione.

### Artefatti

`src/clustering.py` per le variabili per motore, i punteggi, le etichette e la
matrice di aggregazione; `scripts/run_clustering.py` per l'esecuzione. Gli
artefatti stanno in `experiments/clustering/` e non sono versionati.

ESITO: FD003 contiene due popolazioni di traiettorie separabili senza
supervisione, FD001 una sola, e i motori di FD001 sono indistinguibili da una
delle due popolazioni di FD003.

## [10-09-2026] — CORREZIONE: il report separato è escluso, l'argomentazione va nel README

La voce del 27-08, fissando la sequenza dei blocchi, elencava fra i contenuti della fase di
chiusura sia un report sia il README. Ho eliminato il report separato.

Motivo: i due documenti avrebbero occupato lo stesso livello di lettura. Il README è il
documento che chi apre la repository legge per primo, e un secondo file avrebbe duplicato
l'argomentazione oppure costretto il README a rimandare altrove per la parte che lo giustifica.
La divisione che resta è per livello di lettura e non per argomento: i notebook mostrano come si
arriva a un risultato e come si legge, il README dice qual è il risultato e perché è credibile,
con i numeri dentro.

Conseguenza sulla fase: le figure e le tabelle esportate in `results/` sono anche quelle che il
README incorpora, e sono state scelte sapendolo.

## [10-09-2026] — Chiusura del confronto: sesto notebook, figure e tabelle finali

### Il notebook

Ho scritto `notebooks/06_confronto_complessivo.ipynb`. È il documento in cui i quattro blocchi
del confronto e le tre letture di chiusura convergono, e non è un riassunto dei cinque notebook
precedenti, che commentano una famiglia di modelli per volta e non possono dire dove quella
famiglia si collochi rispetto alle altre.

Dieci sezioni: legittimità della composizione, graduatoria complessiva, struttura per famiglie e
sua replica sui due sottoinsiemi, confronto appaiato fold per fold, variabilità dovuta al seme
dello stimatore, trasferimento fuori campione, errore per fascia di vita residua, sensibilità
alla soglia di censura, raggruppamento delle traiettorie, sintesi con i limiti dichiarati.

Il notebook legge artefatti e non esegue lavoro computazionale, come i cinque precedenti, e si
esegue dall'inizio alla fine in pochi secondi.

Motivo della sezione di apertura sulla legittimità: comporre quattro tabelle prodotte da
esecuzioni diverse presuppone partizioni identiche, e il presupposto è verificabile perché le due
baseline sono ricalcolate in ogni blocco. La verifica è la condizione di esistenza di tutto ciò
che segue e sta quindi prima della graduatoria.

### Estensione dello script di raggruppamento

Sintomo: il controllo registrato il 09-09, cioè il raggruppamento ripetuto senza la durata della
traiettoria, non aveva alcun artefatto in `experiments/clustering/` e non era riproducibile
clonando la repository.

Causa: `scripts/run_clustering.py` non calcolava affatto quella variante.

Soluzione: `analyse` riceve un argomento `durations` opzionale, perché il controllo passa una
matrice priva di quella colonna mentre le etichette la riportano comunque, e `main` esegue la
variante dopo l'analisi principale di ciascun sottoinsieme scrivendone punteggi ed etichette. Non
scrivo gli altri tre artefatti della variante: le variabili per motore sarebbero quelle
dell'analisi principale meno una colonna, e la matrice di aggregazione serve al solo dendrogramma
dell'analisi principale. `src/clustering.py` non è stato toccato.

Verifica: la riesecuzione riproduce l'analisi principale su ogni valore registrato il 09-09,
silhouette a due gruppi 0,260 su FD001 e 0,533 su FD003, gruppi di 44 e 56, accordo 1,000 fra
K-Means e Ward su FD003, unione 0,552 con accordo 0,191 con la provenienza. Il conteggio delle
variabili scende di uno nei due blocchi del controllo, quindi la colonna è stata tolta dalla
matrice delle distanze e non solo dalla stampa.

Due fatti che l'esecuzione ha reso visibili e che non erano registrati. L'invarianza di K-Means
al seme a due gruppi vale 1,000 su FD003 ma si ferma a 0,960 su FD001, dove la partizione cambia
leggermente da un seme all'altro: è coerente con l'assenza di gruppi reali, perché senza struttura
il minimo locale trovato dipende dall'inizializzazione. Togliendo la durata, su FD001 l'accordo
fra K-Means e Ward a due gruppi sale da 0,702 a 0,844, mentre la silhouette resta attorno a 0,26 e
decresce dal massimo: la conclusione sull'assenza di struttura non cambia.

### Quantità derivate calcolate nel notebook

Tre quantità riportate nel diario non esistono in alcun artefatto: il conteggio delle inversioni
fra coppie separabili, il divario in dispersioni fra due righe specifiche nei due regimi di
censura, le tabelle incrociate dei raggruppamenti. Le calcola il notebook e le esporta in
`results/tables/`, dove diventano artefatto versionato.

Motivo: la regola del progetto vieta al notebook il lavoro computazionale, non la derivazione da
una tabella di risultati già su disco, e i notebook dei blocchi precedenti già derivano al loro
interno guadagni in dispersioni, piattezza delle griglie ed esponenti empirici di costo.

Alternativa scartata: estendere `scripts/run_holdout.py` e rieseguirlo. Avrebbe comportato
riaddestrare ventidue modelli e due baseline per sottoinsieme e rileggere l'insieme di verifica
ufficiale per persistere un conteggio derivato da una tabella di ventiquattro righe.

Il controllo senza la durata è invece rimasto nello script perché è un riadattamento di K-Means e
non una derivazione: il notebook non lo poteva rifare senza violare la stessa regola.

### Un difetto degli artefatti corretto in lettura

In `{SUBSET}_ranking.csv` e `{SUBSET}_holdout.csv` le due baseline portano l'etichetta del blocco
da cui la graduatoria preleva la copia superstite, cioè `Metodi a margine e reti`. Deriva da
`src/final.py`, che mappa `block` in `blocco` dopo aver scartato le copie duplicate. In una figura
colorata per famiglia avrebbe attribuito il pavimento informativo a una famiglia di modelli.

Soluzione: il notebook assegna alle baseline una categoria propria in lettura. Non ho modificato
`src/final.py`, perché la correzione a monte avrebbe imposto di rigenerare gli artefatti dei
quattro blocchi.

### Concordanza fra graduatoria e insieme di verifica

Il conteggio delle inversioni riguarda le sole coppie che la regola di lettura del progetto
separa in cross-validation, cioè quelle il cui divario supera la dispersione combinata delle due
righe. La correlazione di rango, presa da sola, impone un ordine anche fra righe dichiarate non
ordinabili, e un riordinamento interno a un gruppo indistinguibile la abbassa senza che nulla si
sia invertito.

| Sottoinsieme | Lettura | Coppie separabili su 231 | Inversioni | Correlazione di rango |
|---|---|---|---|---|
| FD001 | tutti i cicli | 136 | 0 | 0,888 |
| FD001 | solo ultimo ciclo | 136 | 0 | 0,984 |
| FD001 | ultimo ciclo, target non censurato | 136 | 0 | 0,981 |
| FD003 | tutti i cicli | 158 | 1 | 0,869 |
| FD003 | solo ultimo ciclo | 158 | 0 | 0,930 |
| FD003 | ultimo ciclo, target non censurato | 158 | 1 | 0,914 |

La correlazione ricalcolata nel notebook riproduce fino all'ultima cifra quella persistita da
`scripts/run_holdout.py`, il che conferma che le due letture operano sugli stessi modelli e sulle
stesse colonne. Le inversioni residue valgono centesimi di ciclo e riguardano righe che la regola
tratta come indistinguibili: l'ordine che il protocollo dichiara leggibile si trasferisce
integralmente su una popolazione indipendente.

Cautela: ogni lettura della verifica è un valore singolo, privo di misura di variabilità, e il
conteggio delle inversioni è descrittivo e fuori dal materiale del corso.

### Errore per fascia di vita residua

Ho scomposto l'errore sull'insieme di verifica per fascia di vita residua vera, usando le
predizioni già persistite e un modello per famiglia scelto sulla graduatoria di FD001, gli stessi
su entrambi i sottoinsiemi. I valori sono in `results/tables/errore_per_vita_residua.csv`.

Il profilo non è monotono. L'errore dei modelli è minimo ai due estremi e ha un massimo nella
fascia intermedia. Nella fascia censurata il target è costante per costruzione e al modello basta
produrre il valore di soglia; quella fascia contiene la maggior parte delle righe di verifica,
quindi la metrica complessiva è pesata verso la parte più facile del problema. Vicino al guasto
l'errore torna basso, ed è la fascia operativamente rilevante. Il massimo cade dove il modello
deve collocare l'inizio del degrado.

La baseline sul solo numero di ciclo ha il profilo opposto, con l'errore massimo nella fascia più
vicina al guasto: una funzione monotona del conteggio non distingue un motore che si guasta presto
da uno che si guasta tardi. La distanza fra i modelli e quella baseline è quindi massima dove il
conteggio fallisce e si riduce nelle fasce intermedie. L'informazione che le letture dei sensori
aggiungono non è distribuita lungo la vita del motore, è concentrata dove il conteggio non basta.

Avevo scritto il commento di questa sezione prima di guardare i valori, prevedendo un errore
massimo nella fascia censurata e decrescente verso il guasto. La forma del profilo lo ha
smentito e ho riscritto il commento sui dati. La scomposizione era una lettura nuova, non
registrata in precedenza, quindi non c'era nulla da cui derivarla.

La scomposizione è descrittiva e non entra in graduatoria: le fasce sono definite sul target vero
e non sono note al momento della predizione.

### Artefatti prodotti

Il notebook esporta dieci figure in `results/figures/` e tredici tabelle in `results/tables/`,
con nomi che non collidono con i file già presenti. La tabella di testa del progetto è
`graduatoria_complessiva.csv`, che riporta per ciascun sottoinsieme media, dispersione, rango e
divario in dispersioni dei ventidue modelli e delle due baseline.

ESITO: la graduatoria del progetto, il confronto appaiato, il diagnostico sul seme, le tre
letture dell'insieme di verifica, la sensibilità alla censura e il raggruppamento delle
traiettorie hanno ora un documento che li legge e artefatti versionati che li registrano. Prima
di questa fase esistevano soltanto in `experiments/`, che non è versionato, e come tabelle in
questo registro.

## [10-09-2026] — CORREZIONE: ampiezza del gruppo non ordinabile nella graduatoria complessiva

La voce del 31-08 riportava che su FD001 sette modelli stanno entro 0,39 cicli e descriveva la
struttura per famiglie come netta e ripetuta identica sui due sottoinsiemi. Applicando alla
graduatoria complessiva la regola di lettura del progetto, cioè il divario inferiore alla
dispersione combinata delle due righe, il gruppo che il protocollo non ordina su FD001 è più
ampio di quei sette e scende fino a comprendere i modelli additivi: dal gruppo di testa restano
separati soltanto l'albero singolo, la famiglia lineare e il kernel polinomiale. Su FD003 lo
stesso criterio si ferma dentro il gradino di testa e lascia fuori i modelli additivi.

I sette modelli entro 0,39 cicli restano il nucleo più stretto della graduatoria e quella
misura non cambia. Ciò che va corretto è la qualificazione della struttura per famiglie. Il
divario fra la famiglia lineare e quella additiva è ampio su entrambi i sottoinsiemi e supera
largamente la dispersione fra fold. Quello fra la famiglia additiva e il gradino di testa è molto
più stretto: resta leggibile su FD003 e non lo è su FD001. Il risultato difendibile è che l'ordine
dei tre gradini si replica sui due sottoinsiemi, mentre la risoluzione con cui il protocollo li
separa cambia da un sottoinsieme all'altro.

La correzione è coerente con il controllo di sensibilità alla censura registrato il 09-09, dove
nel regime censurato il divario del modello additivo dal migliore vale 0,67 dispersioni su FD001
e 1,75 su FD003.

## [10-09-2026] — Chiusura della repository: README, riproducibilità dichiarata e igiene formale

### Il README come unico documento argomentativo

Ho scritto `README.md`, che era l'ultimo artefatto mancante. Con il report
separato già escluso, il README è il documento che porta l'argomentazione, e
deve reggere da solo davanti a un lettore che non ha altre fonti.

Ho scelto un impianto argomentativo invece che cronologico. Motivo: la consegna
chiede di individuare il miglior modello, e la risposta di questo progetto non è
un nome ma una struttura a tre gradini con un vertice non assegnabile su FD001.
Una risposta di questo tipo enunciata in fondo al documento sembra una resa;
enunciata presto e poi difesa è un risultato. Il documento stabilisce quindi
cosa sostiene nella panoramica, espone i dati e il protocollo che rendono
credibili i numeri, presenta la graduatoria, e solo dopo commenta i quattro
blocchi come risposta alla domanda su perché ciascuna famiglia stia dove sta.
Alternativa scartata: l'ordine cronologico che ricalca i sei notebook, che
avrebbe reso il README un loro indice e avrebbe duplicato la spiegazione su due
livelli.

Il commento dei modelli è per famiglia, con ogni modello nominato nel punto in
cui ha un comportamento proprio. Motivo: otto modelli lineari che stanno entro
0,03 cicli non producono otto commenti diversi, e una sezione per ciascuno dei
ventidue avrebbe prodotto ripetizione. La copertura resta verificabile perché
ogni modello ha la sua riga nelle tabelle e una tabella di corrispondenza elenca
i ventidue con il laboratorio di provenienza.

Ho riscritto il documento una seconda volta perché la prima versione, a 719
righe e circa 15.000 parole, era troppo lunga e troppo discorsiva per il suo
lettore, che è un docente di Machine Learning e non ha bisogno delle
spiegazioni divulgative che quella versione conteneva. La versione finale ha 641
righe e circa 12.900 parole, di cui 10.100 di prosa, e incorpora 16 figure sulle
51 disponibili. Il taglio ha riguardato la prosa esplicativa e i racconti estesi
dei problemi tecnici, che restano nel diario. Non ha riguardato le tabelle dei
risultati, i limiti dichiarati accanto al risultato che li produce e i numeri
dentro le affermazioni, che sono la parte che regge una lettura critica.

Limite dichiarato: una decina di numeri citati nel README provengono da misure
registrate qui e non da un artefatto versionato (la quota di righe al valore di
soglia nelle due popolazioni, la curva della perdita della rete, l'impurità
della radice, i tempi della ricerca esaustiva, la tolleranza del controllo
algebrico). Un lettore che apre solo `results/` non può ricontrollarli.

### Decisioni di forma sul README

Niente tempi di esecuzione. Motivo: non ho un registro dei tempi per script, e
un tempo senza la macchina su cui è stato misurato non è informazione. Per la
stessa ragione non compare una macchina di riferimento. La sezione di
riproduzione ordina però gli script per classe di costo relativo, ricavata dai
tempi di ricerca registrati negli artefatti di diagnostica, così che chi clona
sappia quali script durano ore prima di lanciarli.

Nessuna licenza. Motivo: la repository è consegna d'esame e portfolio, non
materiale destinato al riuso, e un file di licenza si aggiunge in qualsiasi
momento senza toccare altro.

### Riproducibilità verificata invece che asserita

Ho ricostruito l'ambiente da zero in un ambiente vuoto con Python 3.12: le 112
versioni fissate in `requirements.txt` si risolvono tutte, senza conflitti e
senza pin irreperibili, e l'installazione completa va a buon fine. Tutti e 17 i
moduli di `src/` si importano, e 14 script su 16 espongono l'interfaccia da riga
di comando. I due che non la espongono, `run_exploration` e `verify_raw_data`,
non hanno argparse per costruzione e falliscono soltanto perché i dati grezzi non
sono presenti, con un messaggio che indica la cartella attesa e rimanda alle
istruzioni di acquisizione.

Limite dichiarato: la ricostruzione è stata verificata su Linux mentre il
progetto gira su macOS, quindi l'insieme delle dipendenze transitive differisce
(su Linux XGBoost tira dentro `nvidia-nccl-cu13`, su macOS richiede invece
`llvm-openmp` da conda, che resta la dipendenza non descritta dal manifesto).
Quello che la verifica stabilisce è che il file di requisiti è risolvibile e
coerente, non che l'ambiente macOS sia riproducibile solo da esso.

### Difetti formali chiusi

Il file di `src/` si chiamava letteralmente `__init__.py ` con uno spazio
finale, quindi il pacchetto non aveva un `__init__.py` valido e funzionava solo
perché Python trattava la cartella come namespace package. Rinominato.

La traslitterazione ASCII degli accenti era più estesa di quanto pensassi: non
riguardava un solo notebook ma quattro (dal 02 al 05, con 54, 59, 72 e 80
occorrenze nella sola prosa) e anche i commenti e le docstring di `src/` e
`scripts/`. Lo stesso spartiacque valeva per il kernel dichiarato nei metadati,
`cmapss-rul` sui notebook 01 e 06 e `python3` sugli altri quattro.

Ho normalizzato con un elenco chiuso di 63 parole italiane, ricavato estraendo
tutti i token vocale più apostrofo presenti nella repository e verificandoli uno
per uno in contesto, invece che con una regola generica. Motivo: una
sostituzione cieca avrebbe colpito le stringhe di codice che finiscono per
apostrofo, come i nomi di colonna `'rmse'` e `'feature'` o i parametri
`'degree'` e `'lasso'`. L'etichetta di blocco `Superamento della linearita'` è
stata esclusa di proposito, perché è un valore scritto dentro cinque tabelle di
`results/` e cambiarla manderebbe codice e artefatti fuori sincrono. Sono 1.752
sostituzioni su 22 file Python e 5 notebook.

La normalizzazione l'ho verificata invece di fidarmene: per ciascuno dei 34 file
Python ho confrontato la sequenza dei token che non sono commenti né stringhe,
identica ovunque, e tutti i file compilano; sui sei notebook ho confrontato
output, contatori di esecuzione e struttura del codice, identici. Nove righe di
codice non commento risultano modificate, e sono titoli di grafico ed etichette
di assi nei notebook 04 e 05: quelle figure cambiano davvero, ed è il
cambiamento voluto.

Il notebook 01 era stato eseguito in un kernel che aveva già eseguito 37 celle,
con contatori da 38 a 56. Tutti e sei i notebook sono stati rieseguiti dal
principio in kernel pulito: contatori da 1 a n e kernel `cmapss-rul` ovunque.

Il commento del `.gitignore` sulla cartella dei dati descriveva una procedura
inesistente, dicendo che i dati sono rigenerabili con lo script di acquisizione,
che era stato valutato e scartato. Riscritto. Le regole le ho verificate con
`git check-ignore` invece di darle per buone. Ho tolto i `.gitkeep` di
`notebooks/`, `scripts/`, `src/`, `results/figures/` e `results/tables/`, ormai
inutili perché quelle cartelle hanno contenuto; restano quelli di `data/` e
`experiments/`, che sono le uniche cartelle vuote per costruzione.

ESITO: la repository contiene sei notebook eseguiti in kernel pulito, 51 figure
e 50 tabelle in `results/`, il README come documento argomentativo
autoconclusivo, e un ambiente dichiarato e verificato.

## [10-09-2026] — CORREZIONE: riferimento lineare sbagliato in guadagno_su_blocco_lineare.csv

Rieseguendo i notebook ho trovato che `results/tables/guadagno_su_blocco_lineare.csv`
cambiava, mentre le altre 49 tabelle restavano identiche. Non era rumore
numerico: cambiava il modello preso come riferimento lineare su FD001, da
`Regressione lineare multipla` a 20,34522 a `Ridge` a 20,32697.

Causa radice: il file versionato era in contraddizione con un altro file
versionato. `FD001_confronto_blocco_lineare.csv` riporta Ridge come miglior
modello lineare di FD001, con divario 0,00, mentre la tabella del guadagno usava
come riferimento la regressione lineare multipla, che in quella stessa tabella è
settima. Il notebook prende come riferimento il miglior modello lineare, e su
FD003 lo faceva correttamente scegliendo forward stepwise. La tabella del
guadagno era quindi stata prodotta quando l'artefatto del blocco lineare di
FD001 era ancora quello incompleto lasciato dalla sovrascrittura parziale, e non
era mai stata rigenerata dopo il ripristino.

Cosa cambia: il guadagno del blocco non lineare su FD001 passa da 2,88 a 2,86
cicli e da 2,41 a 2,44 dispersioni. FD003 non cambia. La conclusione del blocco
non si muove, perché il divario resta ampiamente sopra la soglia di leggibilità
del protocollo, e la struttura per gradini è invariata.

Ho tenuto la versione rigenerata e allineato il numero nel README. Questo rende
superato il valore riportato nella voce sul blocco che supera la linearità.

Nota di metodo: la rigenerazione completa dei notebook in kernel pulito ha fatto
emergere una incoerenza fra artefatti che nessun controllo automatico aveva
intercettato, perché la verifica di identità delle partizioni confronta le
tabelle dei blocchi fra loro e non le tabelle derivate con le tabelle di
partenza.