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