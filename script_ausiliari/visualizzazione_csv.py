import streamlit as st
import pandas as pd
import os

# ------------------------
# Configurazione pagina
# ------------------------
st.set_page_config(layout="wide", page_title="Email Viewer TREC")

# ------------------------
# Funzione per caricare i dati
# ------------------------
@st.cache_data
def load_data():
    file_name = 'dataset_label0.csv'
    
    if not os.path.exists(file_name):
        return None, f"File '{file_name}' non trovato nella cartella."

    try:
        df = pd.read_csv(
            file_name, 
            engine='python', 
            on_bad_lines='skip', 
            encoding='utf-8', 
            encoding_errors='replace'
        )
        return df, None
    except Exception as e:
        return None, str(e)

# ------------------------
# INIZIO APP
# ------------------------
st.title("📧 Visualizzatore Dataset TREC")

# Caricamento dati
with st.spinner('Caricamento del dataset in corso... (può richiedere qualche secondo)'):
    df, error = load_data()

if error:
    st.error(f"Errore: {error}")
    st.warning("Assicurati che il file 'TREC-07.csv' sia nella stessa cartella di questo script.")
elif df is not None and not df.empty:

    # ------------------------
    # Inizializzazione indice email
    # ------------------------
    if "email_index" not in st.session_state:
        st.session_state.email_index = 0

    # ------------------------
    # Sidebar Navigazione
    # ------------------------
    st.sidebar.header("Navigazione")
    st.sidebar.info(f"Email caricate: {len(df)}")

    # Pulsanti Precedente / Successiva
    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("⬅️ Precedente"):
            if st.session_state.email_index > 0:
                st.session_state.email_index -= 1
    with col_next:
        if st.button("➡️ Successiva"):
            if st.session_state.email_index < len(df) - 1:
                st.session_state.email_index += 1

    # Input manuale per l'indice
    manual_index = st.sidebar.number_input(
        "Vai a indice email:", 
        min_value=0, 
        max_value=len(df)-1, 
        value=st.session_state.email_index,
        step=1
    )

    # Aggiorna l'indice se l'utente inserisce manualmente un numero
    st.session_state.email_index = int(manual_index)

    st.sidebar.markdown(
        f"Email corrente: {st.session_state.email_index + 1} / {len(df)}"
    )

    index = st.session_state.email_index

    # ------------------------
    # Estrazione dati riga corrente
    # ------------------------
    row = df.iloc[index]

    sender = row.get('sender', 'Sconosciuto')
    receiver = row.get('receiver', 'Sconosciuto')
    date = row.get('date', 'N/A')
    subject = row.get('subject', '(Nessun oggetto)')
    label = row.get('label', 'N/A')
    body = str(row.get('body', ''))

    # ------------------------
    # Visualizzazione Metadati
    # ------------------------
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.text_input("Da:", value=sender, disabled=True)
    with col2:
        st.text_input("A:", value=receiver, disabled=True)
    with col3:
        st.text_input("Etichetta:", value=label, disabled=True)

    st.markdown(f"### Oggetto: {subject}")
    st.caption(f"Data: {date}")

    st.markdown("---")

    # ------------------------
    # Visualizzazione Body
    # ------------------------
    view_mode = st.radio("Modalità visualizzazione:", ["HTML Renderizzato", "Testo Puro"], horizontal=True)

    if view_mode == "HTML Renderizzato":
        st.components.v1.html(body, height=600, scrolling=True)
    else:
        st.text_area("Contenuto Raw", value=body, height=500)

else:
    st.warning("Il file CSV è vuoto o non è stato possibile leggerlo.")