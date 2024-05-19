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

# Funzione per liberare la memoria
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Funzione per preparare il modello RAG
def prepare_rag_llm(token, vector_store_list, temperature, max_length):
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}  # Usa la CPU
    )

    loaded_db = FAISS.load_local(
        f"vector_store/{vector_store_list}", 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(model="mistral")  # Assicurati che questo modello supporti l'italiano

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory
    )

    return qa_conversation

# Funzione per generare la risposta
@lru_cache(maxsize=128)
def generate_answer(question, token):
    answer = "Si è verificato un errore"
    doc_source = ["nessuna fonte"]

    if token == "":
        answer = "Inserisci il token Hugging Face"
    else:
        response = st.session_state.conversation({"question": question})
        answer = response.get("answer").split("Risposta utile:")[-1].strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

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

    HUGGING_FACE_TOKEN = "nike"
    VECTOR_STORE = "nome_vector_store"
    TEMPERATURE = 1.0
    MAX_LENGTH = 300

    if HUGGING_FACE_TOKEN:
        st.session_state.conversation = prepare_rag_llm(
            HUGGING_FACE_TOKEN, VECTOR_STORE, TEMPERATURE, MAX_LENGTH
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
            answer, doc_source = generate_answer(question, HUGGING_FACE_TOKEN)

        with st.chat_message("assistant"):
            st.markdown(f'<div class="stChatMessage">{answer}</div>', unsafe_allow_html=True)

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    with st.expander("Cronologia della chat e informazioni sui documenti sorgente"):
        st.write(st.session_state.source)

if __name__ == "__main__":
    main()
