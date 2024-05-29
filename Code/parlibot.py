import streamlit as st
import torch
import gc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from googletrans import Translator
from functools import lru_cache

#La funzione clear_memory è utilizzata per liberare 
#la memoria sia dalla GPU (utilizzando PyTorch) che 
#dalla RAM (utilizzando il modulo gc di Python).
def clear_memory():
    # Libera la cache della memoria della GPU utilizzando PyTorch.
    torch.cuda.empty_cache()
    # Effettua la garbage collection manuale per liberare la memoria nella RAM.
    gc.collect()

# Funzione per preparare il modello RAG
#La funzione prepare_rag_llm configura un llm
#per una catena di conversazione basata sul recupero di informazioni. 
#Utilizza un database FAISS per il recupero dei documenti, 
#un modello di embedding per la rappresentazione del testo e un modello di conversazione 
#per gestire le risposte. Questa configurazione è utile per applicazioni di question answering 
#che combinano recupero e generazione di risposte.
def prepare_rag_llm(vector_store_list, temperature, max_length):
    # Inizializza un modello di embedding utilizzando la libreria HuggingFace e specificando di usare la CPU.
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", #modello
        model_kwargs={'device': 'cpu'}  # Usa la CPU
    )
    # Carica un database FAISS locale utilizzando l'instructor_embeddings e permettendo la deserializzazione potenzialmente pericolosa (siamo sicuri dell'integrità dei file).
    loaded_db = FAISS.load_local(
        f"vector_store/{vector_store_list}", 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )

    # Inizializza un modello di linguaggio specificando il modello "llama3".
    llm = ChatOllama(
        model="llama3",
        temperature=temperature,
        num_predict=max_length) 

    # Configura la memoria della conversazione per gestire una finestra di chat history di dimensione 2.
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Configura una catena di conversazione di recupero di informazioni usando il modello di linguaggio e il database FAISS come retriever.
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff", #"Stuff" indica che il testo recuperato viene concatenato
                            #in una singola stringa che viene poi passata al modello di linguaggio per generare la risposta. 
                            #Esistono anche altri tipi di catene, come "map_reduce"(divisione e processamento individuale + ricomposizione e generazione risposta) effettua un processamento parallelo
                            #e "refine"(raffinamento iterativo), che utilizzano strategie ->utile quando le risposte sono critiche
                            #diverse per combinare le informazioni recuperate.
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}), #Configura il retriever per il recupero di documenti. Devono essere recuperati i primi 3 documenti più rilevanti
        return_source_documents=True, #Indica che i documenti sorgente devono essere restituiti insieme alla risposta generata
        memory=memory #history
    )

    # Restituisce la catena di conversazione configurata.
    return qa_conversation

# Funzione per generare la risposta
#La funzione generate_answer è progettata per generare una risposta
#a una domanda utilizzando un modello di conversazione configurato in st.session_state.conversation.
#La funzione verifica la presenza di un token, genera una risposta tramite il modello, 
#traduce la risposta dall'inglese all'italiano e restituisce sia la risposta tradotta che le fonti dei documenti utilizzati per generarla.
#La funzione utilizza una cache LRU (Least Recently Used) per memorizzare fino a 128 risposte generate per migliorare le prestazioni.
@lru_cache(maxsize=128)
def generate_answer(question):
    # Inizializza la risposta predefinita e la fonte del documento
    answer = "Si è verificato un errore"
    doc_source = ["nessuna fonte"]

    # Genera la risposta usando il modello di conversazione
    response = st.session_state.conversation({"question": question})
    
    # Estrae la risposta dal response
    answer = response.get("answer").split("Risposta utile:")[-1].strip()
    
    # Estrae le fonti dei documenti dal response
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]

    # Traduttore per tradurre la risposta in italiano
    translator = Translator()
    translated_answer = translator.translate(answer, src='en', dest='it').text

    # Restituisce la risposta tradotta e le fonti dei documenti
    return translated_answer, doc_source

# Funzione principale
def main():
    clear_memory()

    st.image("/Users/nikedigiacomo/Desktop/logo.png", width=200)

    # Caricamento del file CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

    VECTOR_STORE = "nome_vector_store"
    TEMPERATURE = 1.0
    MAX_LENGTH = 300

    st.session_state.conversation = prepare_rag_llm(
        VECTOR_STORE, TEMPERATURE, MAX_LENGTH
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
            answer, doc_source = generate_answer(question)

        with st.chat_message("assistant"):
            st.markdown(f'<div class="stChatMessage">{answer}</div>', unsafe_allow_html=True)

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    with st.expander("Cronologia della chat e informazioni sui documenti sorgente"):
        st.write(st.session_state.source)

if __name__ == "__main__":
    main()
