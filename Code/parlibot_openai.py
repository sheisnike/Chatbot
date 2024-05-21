import streamlit as st
import openai
import gc
from googletrans import Translator
from functools import lru_cache
from dotenv import load_dotenv
import os

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Imposta la tua chiave API di OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Funzione per liberare la memoria
def clear_memory():
    gc.collect()

# Funzione per preparare il modello RAG (non necessaria con OpenAI)
def prepare_rag_llm():
    return None  # Non è più necessario preparare un modello locale

# Funzione per generare la risposta usando OpenAI
@lru_cache(maxsize=128)
def generate_answer(question, token):
    answer = "Si è verificato un errore"
    doc_source = ["nessuna fonte"]

    if token == "":
        answer = "Inserisci il token OpenAI"
    else:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            )
            answer = response['choices'][0]['message']['content'].strip()
            doc_source = ["Nessuna fonte disponibile per i modelli GPT"]

            translator = Translator()
            translated_answer = translator.translate(answer, src='en', dest='it').text

            return translated_answer, doc_source
        except Exception as e:
            return str(e), doc_source

# Funzione principale
def main():
    clear_memory()

    st.image("/Users/nikedigiacomo/Desktop/logo.png", width=200)

    # Definizione dello stile
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .main { background-color: #e8f5e9; color: #2e7d32; }
    .stButton>button { background-color: #66bb6a; color: white; }
    .stTitle { font-family: 'Roboto', sans-serif; font-size: 2.5em; color: #1b5e20; text-align: center; margin-bottom: 0; }
    .stHeader { font-family: 'Roboto', sans-serif; font-size: 1.5em; color: #1b5e20; }
    .stExpander { font-family: 'Roboto', sans-serif; font-size: 1em; color: #1b5e20; }
    .stMarkdown { font-family: 'Roboto', sans-serif; color: #2e7d32; }
    .stChatMessage { background-color: #ffffff; padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #cccccc; }
    .stChatInput { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='stTitle'>ParliBot 🤖</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-family: "Roboto", sans-serif; font-size: 1.2em; color: #1b5e20;'>
            Benvenuto nel Chatbot RAG! Interagisci con il Deputato Digitale!
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "source" not in st.session_state:
        st.session_state.source = []

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    st.subheader("Cronologia della Chat")
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="stChatMessage">{message["content"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Fai una domanda")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(f'<div class="stChatMessage">{question}</div>', unsafe_allow_html=True)

        # Aggiungi una barra di progresso durante l'elaborazione della risposta
        with st.spinner('Sto elaborando la tua risposta...'):
            answer, doc_source = generate_answer(question, OPENAI_API_KEY)

        with st.chat_message("assistant"):
            st.markdown(f'<div class="stChatMessage">{answer}</div>', unsafe_allow_html=True)

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    with st.expander("Cronologia della chat e informazioni sui documenti sorgente"):
        st.write(st.session_state.source)

if __name__ == "__main__":
    main()
