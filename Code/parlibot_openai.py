import streamlit as st
import gc
import os
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from googletrans import Translator
from functools import lru_cache
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Funzione per liberare la memoria
def clear_memory():

    #Effettua la garbage collection per liberare la memoria nella RAM
    gc.collect()

# Funzione per preparare il modello RAG
def prepare_rag_llm(api_key, vector_store_list, temperature, max_length):

    #Inizializza un modello di embedding basato su OpenAI.
    openai_embeddings = OpenAIEmbeddings(api_key=api_key)

    #Carica il database FAISS utilizzando gli embeddings generati da OpenAI e permettendo la deserializzazione potenzialmente pericolosa
    loaded_db = FAISS.load_local(
        f"vector_store/{vector_store_list}", 
        openai_embeddings, 
        allow_dangerous_deserialization=True
    )

    #Inizializza il modello di linguaggio specificando il modello "gpt-3.5", la api key di OpenAI, la temperatura e il numero massimo di token da generare
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  #Utilizza come modello di llm gpt-3.5-turbo
        api_key=api_key, #Imposta la API key di OpenAI fornita in ingresso
        temperature=temperature,
        max_tokens=max_length
    )

    #Configura la memoria della conversazione per gestire una finestra di dimensione 2
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    #Configura la catena di conversazione di recupero delle informazioni utilizzando il modello di OpenAI impostato e il database FAISS 
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory
    )

    #Restituisce la catena di conversazione
    return qa_conversation

# Funzione per generare la risposta
@lru_cache(maxsize=128)
def generate_answer(question, api_key):
    answer = "Si è verificato un errore"
    doc_source = ["nessuna fonte"]

    if api_key == "":
        answer = "Inserisci il token OpenAI"
    else:

        #Genera la risposta usando il modello di conversazione
        response = st.session_state.conversation({"question": question})
        
        #Estrae la risposta dal response
        answer = response.get("answer").split("Risposta utile:")[-1].strip()
        
        #Estrare le fonti dei documenti dal response
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

    #Traduzione della risposta in italiano
    translator = Translator()
    translated_answer = translator.translate(answer, src='en', dest='it').text

    return translated_answer, doc_source

# Funzione principale
def main():
    clear_memory()

    st.image("/Users/nikedigiacomo/Desktop/logo.png", width=200)

    # Definizione dello stile
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    .main {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #66bb6a;
        color: white;
    }
    .stTitle {
        font-family: 'Roboto', sans-serif;
        font-size: 2.5em;
        color: #1b5e20;
        text-align: center;
        margin-bottom: 0;
    }
    .stHeader {
        font-family: 'Roboto', sans-serif;
        font-size: 1.5em;
        color: #1b5e20;
    }
    .stExpander {
        font-family: 'Roboto', sans-serif;
        font-size: 1em;
        color: #1b5e20;
    }
    .stMarkdown {
        font-family: 'Roboto', sans-serif;
        color: #2e7d32;
    }
    .stChatMessage {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #cccccc;
    }
    .stChatInput {
        background-color: #ffffff;
    }
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

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "source" not in st.session_state:
        st.session_state.source = []

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #Imposta la chiave di OpenAI prendendola dal file .env
    VECTOR_STORE = "nome_vector_store"
    TEMPERATURE = 1.0
    MAX_LENGTH = 300

    if OPENAI_API_KEY:
        st.session_state.conversation = prepare_rag_llm(
            OPENAI_API_KEY, VECTOR_STORE, TEMPERATURE, MAX_LENGTH
        )

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
