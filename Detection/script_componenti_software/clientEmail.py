# ==============================
# IMPORT DELLE LIBRERIE
# ==============================

import imaplib              # Gestione connessione server via protocollo IMAP
import email                # Parsing e decodifica dei messaggi email 
import time                 # Gestione dei cicli di attesa (sleep) per il polling
import torch                # Backend PyTorch per l'esecuzione del modello neurale
from transformers import AutoModelForCausalLM, AutoTokenizer  # Classi base per caricare modello e tokenizzatore
from peft import PeftModel  # Integrazione PEFT per caricare adapter LoRA/QLoRA
from huggingface_hub import login  # Client per l'autenticazione su Hugging Face

def load_model():
    """
    Inizializza l'ambiente di inferenza caricando il modello linguistico basato su LLaMA-2 fine-tuned tramite LoRA 
    e le configurazioni necessarie.
    
    Il processo include:
    1. Autenticazione presso l'Hub di Hugging Face.
    2. Caricamento del tokenizzatore dal checkpoint fine-tuned.
    3. Caricamento del modello base (LLaMA-2) con quantizzazione a 8-bit per l'efficienza.
    4. Integrazione dell'adapter LoRA (Low-Rank Adaptation) per la specializzazione del modello.
    5. Configurazione del modello in modalità di valutazione (inference-only).

    Returns:
        tuple: Una coppia (tokenizer, model) pronta per l'inferenza
    """

    # Autenticazione Hugging Face
    # login(token='')  

    # ----------------------------
    # CONFIGURAZIONE MODELLO
    # ----------------------------

    ## Identificativo del modello foundation LLaMA-2 disponibile su Hugging Face
    BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

    # Percorso locale contenente i pesi dell'adapter LoRA fine-tuned
    LORA_ADAPTER_PATH = "./llama-finetuned-final"

    # ----------------------------
    # CARICAMENTO TOKENIZER
    # ----------------------------

    # Carica il tokenizzatore associato ai pesi fine-tuned
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

    # LLaMA non definisce un padding token di default.
    # Per evitare errori durante l'inferenza batch-based, si imposta
    # il token di fine sequenza (EOS) come padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # CARICAMENTO MODELLO BASE
    # ----------------------------

    # Carica il modello LLaMA-2 in modalità quantizzata a 8 bit.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_8bit=True,    # Abilita la quantizzazione int8 
        device_map="auto"     # Assegna automaticamente i layer ai dispositivi disponibili
    )

    # ----------------------------
    # CARICAMENTO ADAPTER LoRA
    # ----------------------------

    # Applica i pesi dell'adapter LoRA al modello base congelato
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH
    )

    # Imposta il modello in modalità inferenza (disabilita il dropout ed evita il calcolo gradienti)
    model.eval()

    return tokenizer, model



# ==============================
# FUNZIONE DI TEST DEL MODELLO
# ==============================

def try_prediction(model,tokenizer):    

    """
    Esegue un test di inferenza su un esempio statico.
    
    
    Permette di cerificare che la pipeline (modello + tokenizer) sia caricata correttamente
    e che il modello risponda rispettando il formato di classificazione binaria (0/1)
    appreso durante il fine-tuning.
    
    Args:
        model: Il modello LLM caricato.
        tokenizer: Il tokenizzatore associato al modello.
    """

    # -------------------------------------------------------
    # 1. DEFINIZIONE DEL PROMPT 
    # -------------------------------------------------------

    # Il prompt segue il formato Instruction-Input-Response,
    # identico a quello utilizzato durante la fase di addestramento.
    prompt = """### Instruction:
You are a classification model specializing in emails, and your job is to detect phishing: respond only with \"0\" if it is not phishing or \"1\" if it is phishing, without explanations, symbols, additional letters, or other characters.
### Input:
We attempted to deliver your package today, but were unable to complete the delivery due to missing address information.
Please update your delivery details as soon as possible to avoid return of the shipment:
Update Delivery Information
Thank you for your cooperation,
Logistics Service Team
### Response:
"""

    # -------------------------------------------------------
    # 2. PRE-PROCESSING (TOKENIZZAZIONE)
    # -------------------------------------------------------
    
    enc = tokenizer(
        prompt,
        return_tensors="pt", # abilita la compatibilità con il backend PyTorch,
        padding=True, # garantisce dimensioni consistenti in caso di batch.
    )


    # Trasferimento esplicito dei tensori sulla GPU.
    # (Necessario affinché i dati di input siano sullo stesso device del modello)
    input_ids = enc.input_ids.cuda()

    # Trasferisce la maschera di attenzione sulla GPU.
    attention_mask = enc.attention_mask.cuda()


    # -------------------------------------------------------
    # 3. INFERENZA (GENERAZIONE)
    # -------------------------------------------------------

    # Genera la risposta del modello.
    outputs = model.generate(
        input_ids=input_ids, # contiene la sequenza di token che rappresenta il prompt
                             # e costituisce il contesto iniziale per la generazione del modello.
        attention_mask=attention_mask, # specifica quali token devono essere considerati durante il calcolo dell'attenzione, escludendo i token di padding.
        max_new_tokens=1, # Forziamo il modello a generare un solo token, 
                          # poiché ci aspettiamo esclusivamente una classe binaria ("0" o "1").
        pad_token_id=tokenizer.eos_token_id # specifica il token utilizzato per il padding.
    )


    # -------------------------------------------------------
    # 4. POST-PROCESSING (DECODING)
    # -------------------------------------------------------

    # L'output del modello, espresso come tensore di token,
    # viene decodificato in una stringa leggibile.
    # L'opzione skip_special_tokens=True rimuove eventuali token
    # speciali introdotti dal modello (<s>, </s>, ecc.).  
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   


# ==============================================================================
# UTILITY DI POST-PROCESSING E VISUALIZZAZIONE
# ==============================================================================


def print_prediction(risultato):

    """
    Interpreta l'output grezzo del modello e visualizza un report formattato per l'utente.

    La funzione esegue il mapping tra l'output numerico e le classi semantiche:
    - 0: Classe Negativa (Email Legittima/Ham)
    - 1: Classe Positiva (Phishing)

    Include una gestione degli errori per output non conformi (es. se il modello genera testo
    invece di token numerici).
    
    """

    print("\n🔍 Analisi email completata")
    print("────────────────────────────")
    
    # -------------------------------------------------------
    # VALIDAZIONE E CASTING DELL'OUTPUT
    # -------------------------------------------------------
    
    try:

        # Tenta la conversione dell'output del modello in un intero.
        # Questo passaggio svolge una duplice funzione:
        # 1) Converte il risultato in un formato numerico utilizzabile.
        # 2) Valida implicitamente l'output, sollevando un'eccezione
        #    nel caso in cui il modello abbia generato testo non numerico.
        risultato = int(risultato) 

        # -------------------------------------------------------
        # LOGICA DI CLASSIFICAZIONE BINARIA
        # -------------------------------------------------------

        # Verifica se la classe predetta corrisponde a 0.
        # In questo caso l'email viene considerata legittima.
        if risultato == 0:

            # Stampa un messaggio che indica l'assenza di minacce.
            print("✅ ESITO: EMAIL LEGITTIMA")

            # Visualizza il valore numerico restituito dal modello
            print("📊 Predizione modello:", risultato)

        else:
            # Gestisce il caso in cui il risultato sia diverso da 0.
            # Anche se il modello è addestrato su classi binarie (0/1),
            # qualsiasi valore non nullo viene trattato come phishing
            # per adottare un approccio conservativo alla sicurezza.
            print("🚨 ESITO: EMAIL DI PHISHING")

            # Stampa il valore numerico generato dal modello.
            print("📊 Predizione modello:", risultato)

        print("────────────────────────────\n")

    # -------------------------------------------------------
    # GESTIONE ERRORI / OUTPUT NON CONFORMI
    # -------------------------------------------------------

    except:
        
        # Questo blocco intercetta tutte le eccezioni sollevate
        # durante il casting a intero (es. ValueError).
        # Tali casi possono verificarsi se il modello genera
        # testo discorsivo o caratteri non previsti (hallucinations).
        
        # Stampa un messaggio di avviso che segnala
        # l'impossibilità di fornire una classificazione affidabile.
        print("\n⚠️  Impossibile fornire un esito affidabile")

        # Visualizza l'output originale del modello 
        print("📊 Output del modello:", risultato)

        print("────────────────────────────\n")



def clean_file(output_file):
    
    """
    Inizializza o resetta il file di report eliminandone il contenuto precedente.

    La funzione apre il file specificato in modalità di scrittura ('w'),
    operazione che tronca automaticamente il file a lunghezza zero.
    Questo approccio garantisce che ogni nuova esecuzione del sistema
    parta da un file di log pulito e privo di dati residui.
    
    """

    # Apre il file di output in modalità scrittura ('w').
    # Se il file esiste, il suo contenuto viene eliminato;
    # se non esiste, viene creato ex novo.
    with open(output_file, "w", encoding="utf-8") as f:

        # L'istruzione 'pass' indica esplicitamente l'assenza di operazioni.
        pass  



def write_report(fields, metadata_list, body, prediction,output_file):

    """
    Genera un report strutturato relativo a una singola email analizzata
    e lo registra in modo persistente su file.

    Il report include:
    - metadati dell'email;
    - corpo testuale del messaggio;
    - esito della classificazione effettuata dal modello.

    Args:
        fields: Etichette dei metadati.
        metadata_list: Valori estratti corrispondenti ai campi.
        body: Il corpo del testo dell'email.
        prediction: L'output grezzo del modello ("0", "1").
        output_file: Path del file di destinazione.
    """

    # -------------------------------------------------------
    # 1. MAPPING PREDITTORE -> ETICHETTA LEGGIBILE
    # -------------------------------------------------------

    # Se il modello restituisce "1", l'email viene classificata come phishing.
    if prediction == "1":
        result_text = "PHISHING"
    
    # Se il modello restituisce "0", l'email viene classificata come legittima.
    elif prediction == "0":
        result_text = "NON PHISHING"
    
    # Gestisce eventuali casi anomali in cui l'output del modello
    # non rispetti il formato binario atteso (es. errori di generazione).
    else:
        result_text = "IMPOSSIBILE DETERMINARE LA PREDIZIONE"
        

    # -------------------------------------------------------
    # 2. SCRITTURA SU FILE (APPEND MODE)
    # -------------------------------------------------------

    # Apre il file di output in modalità append ('a') per accodare
    # il nuovo report senza sovrascrivere quelli già presenti.
    with open(output_file, "a", encoding="utf-8") as f:

        # Scrive un'intestazione che delimita chiaramente l'inizio
        # di un nuovo report relativo a una singola email.
        f.write("===== EMAIL REPORT =====\n")

        # Scrittura dinamica dei metadati (Sender, Receiver, Date, Subject)
        f.write(f"{fields[0]}    : {metadata_list[0]}\n")
        f.write(f"{fields[1]}    : {metadata_list[1]}\n")
        f.write(f"{fields[2]}    : {metadata_list[2]}\n")
        f.write(f"{fields[3]}    : {metadata_list[3]}\n\n")

        # Scrive il contenuto dell'email così come estratto,
        # preservandone la formattazione testuale.
        f.write("----- BODY -----\n")
        f.write(body + "\n\n")

        # Registra l'esito della classificazione effettuata dal modello,
        # includendo sia l'etichetta semantica sia l'output numerico grezzo.
        f.write(f"Prediction: {result_text} ({prediction})\n")
        f.write("========================\n\n")


# ==============================================================================
# GESTIONE CONNESSIONE E AUTENTICAZIONE (IMAP)
# ==============================================================================

def connect(EMAIL,APP_PASSWORD):

    """
    Stabilisce una connessione IMAP sicura con i server di posta di Gmail
    e restituisce una sessione autenticata pronta per l'interrogazione
    delle email.

    La connessione utilizza il protocollo IMAP su SSL/TLS al fine di
    garantire la riservatezza delle credenziali e dei dati scambiati.

    Args:
        EMAIL (str): indirizzo email dell'account da monitorare.
        APP_PASSWORD (str): password per le applicazioni generata da Google,
                            necessaria per l'accesso IMAP in ambienti sicuri.

    Returns:
        imaplib.IMAP4_SSL: oggetto rappresentante la sessione IMAP autenticata.

    Raises:
        RuntimeError: sollevata se la selezione della mailbox fallisce.
    """


    # -------------------------------------------------------
    # 1. HANDSHAKE E CONNESSIONE SSL
    # -------------------------------------------------------

    # Crea un'istanza del client IMAP utilizzando una connessione SSL/TLS.
    mail = imaplib.IMAP4_SSL("imap.gmail.com")

    # -------------------------------------------------------
    # 2. AUTENTICAZIONE
    # -------------------------------------------------------

     # Esegue il login presso il server IMAP utilizzando le credenziali fornite.
    mail.login(EMAIL, APP_PASSWORD)

    # -------------------------------------------------------
    # 3. SELEZIONE DEL CONTESTO (MAILBOX)
    # -------------------------------------------------------
    
    # Seleziona la cartella "[Gmail]/Tutti i messaggi", che contiene
    # l'intero archivio delle email dell'account, indipendentemente
    # dalla loro classificazione (in arrivo, inviate, archiviate, ecc.).
    status, _ = mail.select('"[Gmail]/Tutti i messaggi"')

    # Verifica che la selezione della mailbox sia avvenuta correttamente.
    # Il server restituisce lo stato "OK" in caso di successo.
    if status != "OK":
        # In caso di fallimento, viene sollevata un'eccezione per segnalare
        # l'impossibilità di procedere con il polling delle email.
        raise RuntimeError("Mailbox non selezionata")

    # Restituisce l'oggetto della sessione IMAP autenticata e configurata
    return mail

# ==============================================================================
# PIPELINE DI INFERENZA (PREDIZIONE)
# ==============================================================================

def do_prediction(body,model,tokenizer):       

    """
    
    Esegue la classificazione di un singolo corpo email tramite
    un modello linguistico fine-tuned per il rilevamento di phishing.

    La funzione implementa l'intera pipeline di inferenza:
    - costruzione del prompt;
    - tokenizzazione e controlli sulla lunghezza;
    - inferenza tramite generazione controllata;
    - parsing dell'output del modello.

    Args:
        body: contenuto testuale dell'email da analizzare.
        model: modello LLM caricato in memoria (modalità inference).
        tokenizer: tokenizzatore associato al modello.

    Returns:
        str: stringa rappresentante la classe predetta:
             "0" → email legittima,
             "1" → email di phishing,
             oppure un messaggio di errore in caso di input non valido.
    
    """
    
    # -------------------------------------------------------
    # 1. DEFINIZIONE VINCOLI (CONTEXT WINDOW)
    # -------------------------------------------------------
    
    # Definisce il numero massimo di token consentiti per l'input.
    dim_max = 1024      


    # -------------------------------------------------------
    # 2. PROMPT ENGINEERING
    # -------------------------------------------------------

    # Costruisce il prompt utilizzando il formato Instruction-Input-Response.
    prompt = f"""
### Instruction:
You are a classification model specializing in emails, and your job is to detect phishing: respond only with \"0\" if it is not phishing or \"1\" if it is phishing, without explanations, symbols, additional letters, or other characters.
### Input:
{body}
### Response:
"""

    # -------------------------------------------------------
    # 3. PRE-PROCESSING E CONTROLLI
    # -------------------------------------------------------

    # Tokenizza il prompt e lo converte in tensori PyTorch.
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    )

     # Estrae la lunghezza della sequenza tokenizzata.
    num_tokens = enc.input_ids.shape[1]

    # Verifica che la lunghezza della sequenza non superi
    # il limite massimo consentito.
    if num_tokens > dim_max:
        return "Formato Sbagliato" 

    # Trasferisce i tensori di input dalla CPU alla GPU.
    input_ids = enc.input_ids.cuda()

    # Trasferisce l'attention mask dalla CPU alla GPU.
    attention_mask = enc.attention_mask.cuda() 

    # -------------------------------------------------------
    # 4. GENERAZIONE (INFERENZA)
    # -------------------------------------------------------

    # Esegue la generazione dell'output del modello.
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id
    )


    # -------------------------------------------------------
    # 5. POST-PROCESSING E PARSING
    # -------------------------------------------------------

    # Decodifica la sequenza generata dal modello in formato testuale.
    # L'output include sia il prompt originale sia il token generato.
    risposta = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Estrae esclusivamente la porzione di testo successiva al tag
    # "### Response:", che rappresenta l'effettiva predizione del modello.
    risultato= risposta.split("### Response:", 1)[1].strip()  

    # Restituisce la predizione finale ("0" o "1") come stringa.
    return risultato
    

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


# Carica in memoria VRAM il modello LLM e il tokenizzatore associato.
tokenizer,model = load_model()


# Credenziali per l'accesso al server IMAP di Gmail.
EMAIL = "labexp2526@gmail.com"                             
APP_PASSWORD = "sqpn ljaw eidd duqj".replace(" ", "")       

# Definizione dello schema dei metadati da estrarre per la reportistica.
FIELDS = ["From", "To", "Date", "Subject"]

# Percorso del file di log per la persistenza dei risultati delle predizioni.
output_file = "report_email.txt"


# -------------------------------------------------------
# SETUP CONNESSIONE E STATO
# -------------------------------------------------------

# Connessione cifrata (SSL) con il provider di posta Gmail.
mail = connect(EMAIL,APP_PASSWORD)

# Registro dei messaggi già processati, memorizzati tramite UID.
seen_uids = set()

# ==============================
# LETTURA DELLE EMAIL ESISTENTI
# ==============================

# Loop per acquisire la modalità operativa dell'utente.
while True:

    # Inserimento modalità: 1 = analisi email passate, 0 = solo nuove email.
    flag_input = input("Inserisci 1 per leggere tutte le email passate, 0 per leggere solo le nuove email: ").strip()
    
    if flag_input in ("0", "1"):
        # Imposta la strategia di lettura.
        flag_past_email = int(flag_input)
        break
    else:
        # Gestione input non valido
        print("Input non valido. Inserisci solo 0 o 1.")

# Flag per gestire la scrittura del report (header del file).
# 0 significa che l'header non è ancora stato controllato in questa sessione.
flag_report = 0

# Loop infinito per acquisire input corretto dall'utente
# relativo alla modalità di reportistica (verbose o non verbose)
while True:

    # Richiesta input all'utente:
    # 1 -> stampa la predizione di ogni email sul terminale
    # 0 -> non stampare predizioni (modalità silenziosa)
    flag_input = input("Inserisci 1 per stampare la predizione di ogni email analizzata, 0 altrimenti: ").strip()
    
    # Controllo validità dell'input: accetta solo "0" o "1"
    if flag_input in ("0", "1"):
        
        # Conversione dell'input da stringa a intero
        # flag_report fungerà da flag booleano (0 = no verbose, 1 = verbose)
        flag_report = int(flag_input)
        break
    else:
        # Messaggio di errore in caso di input non valido
        # Il loop ricomincia finché l'utente non inserisce un valore corretto
        print("Input non valido. Inserisci solo 0 o 1.")

# Se il reporting è attivo, resetta il file di log.
if flag_report == 1:
    clean_file(output_file)

# -------------------------------------------------------
# RECUPERO DELLE EMAIL PASSATE
# -------------------------------------------------------

# Se l'utente ha scelto di analizzare lo storico 
if flag_past_email == 1:

    # Esegue il comando IMAP 'UID SEARCH' per ottenere tutti gli UID dei messaggi
    # presenti nella mailbox. "ALL" indica che vogliamo l'elenco completo.
    status, data = mail.uid("search", None, "ALL")

    # Converte la risposta in una lista di UID.
    # La risposta è una lista di byte, quindi split() produce una lista di UID separati.
    uids = data[0].split()

    # Stampa il numero totale di email presenti sul server.
    print(f"📬 Mail presenti: {len(uids)}")

    # ==============================================================================
    # CICLO DI ELABORAZIONE BATCH
    # ==============================================================================

    for uid in uids:

        # -------------------------------------------------------
        # CICLO DI ELABORAZIONE EMAIL PASSATE
        # -------------------------------------------------------

        # Aggiunge l'UID al set 'seen_uids' per evitare doppie elaborazioni nelle fasi successive 
        seen_uids.add(int(uid))  
    
         # Richiede il messaggio completo al server (RFC822 = header + body)
        status, msg_data = mail.uid("fetch", uid, "(RFC822)")

        # Converte i byte ricevuti in oggetto email leggibile da Python
        # tramite la libreria 'email'.
        msg = email.message_from_bytes(msg_data[0][1])
        
        # -------------------------------------------------------
        # ESTRAZIONE METADATI 
        # -------------------------------------------------------

        metadata_list=[]
        print("\n============================")
        for f in FIELDS:

            # Estrazione dei metadati specifici (From, To, Date, Subject)
            # msg.get(f) gestisce automaticamente la decodifica RFC2047
            # dei caratteri speciali negli header delle email.
            print(f"{f:<10}: {msg.get(f)}")
            metadata_list.append(msg.get(f))

        # -------------------------------------------------------
        # PARSING DEL CORPO (MIME TRAVERSAL)
        # -------------------------------------------------------

    
        body = ""

        # msg.walk() itera ricorsivamente su tutte le parti del messaggio.
        for part in msg.walk():
            
            # Filtra solo le parti 'text/plain' e ignora allegati
            # verificando l'assenza di 'Content-Disposition'.
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                
                # Determina la codifica dei caratteri della parte.
                # Se non specificata, fallback a 'utf-8'.
                charset = part.get_content_charset() or "utf-8"

                # Decodifica del payload (byte -> stringa).
                # errors="ignore" previene crash dovuti a caratteri malformati.
                body = part.get_payload(decode=True).decode(charset, errors="ignore")

                print("\n--- BODY ---")
                print(body)

        
        # Concateniamo i metadati al corpo del messaggio.
        metadata_prompt = "sender: "+str(metadata_list[0])+" | "+"receiver: "+str(metadata_list[1])+" | "+"date: "+str(metadata_list[2])+" | "+"subject: "+str(metadata_list[3])
        
        # Costruzione stringa finale per il LLM
        body_input = metadata_prompt+" | "+"body: "+body
        
        # Esecuzione della pipeline di predizione (Tokenizzazione -> Modello -> Output)
        risultato = do_prediction(body_input,model,tokenizer)

        # Visualizzazione risultato in console
        print_prediction(risultato)     

        # Creazione del report se richiesto
        if flag_report == 1:
            write_report(FIELDS, metadata_list, body, risultato, output_file)

# Segnalazione termine fase batch
print("\nIn ascolto per nuove email...")

# ==============================================================================
# INIZIALIZZAZIONE MONITORAGGIO REAL-TIME (BASELINE)
# ==============================================================================

# Riconnessione al server 
mail = connect(EMAIL,APP_PASSWORD)

# Interroghiamo il server per ottenere la lista completa degli UID attualmente presenti.
status, data = mail.uid("search", None, "ALL")

# Converte tutti gli UID delle email presenti sul server in interi e li salva in un set
# per tenere traccia delle email già processate (evita duplicati e permette controlli rapidi).
seen_uids = set(int(uid) for uid in data[0].split())


# ==============================================================================
# MONITORAGGIO REAL-TIME (POLLING LOOP)
# ==============================================================================

# Ciclo infinito per il monitoraggio continuo.
# Il sistema interroga periodicamente il server per rilevare nuovi arrivi.
while True:
    try:

        # -------------------------------------------------------
        # KEEP-ALIVE E SINCRONIZZAZIONE
        # -------------------------------------------------------

        # Invia un comando 'NOOP' (No Operation) al server.
        # Scopo: Mantenere attiva la sessione TCP/IMAP prevenendo il timeout
        # del server dovuto a inattività (Heartbeat mechanism).
        mail.noop()

        # Recupera lo stato corrente della mailbox inviando 'UID SEARCH ALL'
        # Otteniamo tutti gli UID dei messaggi presenti sul server.
        status, data = mail.uid("search", None, "ALL")

        # Parsing della risposta: converte tutti gli UID in interi
        uids = [int(uid) for uid in data[0].split()]

        # Identifica i nuovi messaggi confrontando gli UID correnti
        # con quelli già processati e memorizzati in 'seen_uids'.
        new_uids = [uid for uid in uids if uid not in seen_uids]
        
        # -------------------------------------------------------
        # ELABORAZIONE NUOVI MESSAGGI
        # -------------------------------------------------------

    
        # Ciclo sui nuovi UID identificati
        for uid in new_uids:

            # Fetching del messaggio completo (Header + Body)
            status, msg_data = mail.uid("fetch", str(uid), "(RFC822)")

            # Parsing in oggetto email leggibile da Python
            msg = email.message_from_bytes(msg_data[0][1])
            
            # Inizializzazione della lista dei metadati (From, To, Date, Subject)
            metadata_list=[]
            print("\n📩 NUOVA MAIL")

            # Ciclo sui campi di interesse definiti in FIELDS
            for f in FIELDS:
                # Stampa il campo e lo aggiunge alla lista dei metadati
                # msg.get(f) gestisce automaticamente eventuali codifiche speciali RFC2047
                print(f"{f:<10}: {msg.get(f)}")
                metadata_list.append(msg.get(f))

            body = ""
            
            # Itera su tutte le parti del messaggio
            for part in msg.walk():
                # Considera solo testo semplice (text/plain)
                # Ignora allegati e parti HTML verificando 'Content-Disposition'
                if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):

                    # Determina la codifica dei caratteri della parte
                    charset = part.get_content_charset() or "utf-8"

                    # Decodifica da byte a stringa
                    # errors="ignore" evita crash in caso di caratteri malformati
                    body = part.get_payload(decode=True).decode(charset, errors="ignore")

                    # Stampa il corpo dell'email
                    print("\n--- BODY ---")
                    print(body)

            # Aggiorna 'seen_uids' subito dopo l'elaborazione
            seen_uids.add(uid)

            # Concatenazione dei metadati al corpo della mail
            metadata_prompt = "sender: "+str(metadata_list[0])+" | "+"receiver: "+str(metadata_list[1])+" | "+"date: "+str(metadata_list[2])+" | "+"subject: "+str(metadata_list[3])
            
            # Prompt finale per il modello
            body_input = metadata_prompt+" | "+"body: "+body
            #print(body_input)

            # Esegue la pipeline completa: tokenizzazione -> modello -> output
            risultato = do_prediction(body_input,model,tokenizer)

            # Visualizza il risultato della predizione in console
            print_prediction(risultato)    

            
            # Se la modalità verbose/report è attiva, scrive il risultato su file
            if flag_report == 1:
                write_report(FIELDS, metadata_list, body, risultato, output_file)



    # -------------------------------------------------------
    # GESTIONE ECCEZIONI E RICONNESSIONE
    # -------------------------------------------------------

    except imaplib.IMAP4.abort:
        # Gestisce la disconnessione improvvisa dal server (es. reset della connessione).
        # Invece di crashare, il sistema tenta di ristabilire la sessione.
        print("🔄 Connessione persa, riconnessione...")
        mail = connect(EMAIL,APP_PASSWORD)

    
    # -------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------

    # Pausa operativa per ridurre il carico sulla CPU e rispettare le policy
    # di frequenza richieste del server IMAP.
    time.sleep(1)