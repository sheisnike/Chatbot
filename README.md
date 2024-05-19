# ParliBot 🤖

**ParliBot** è un chatbot interattivo basato su un modello di Retrieval-Augmented Generation (RAG), progettato per rispondere a domande basate su documenti PDF uniti e indicizzati. Questo progetto utilizza tecnologie di NLP avanzate per fornire risposte accurate e contestuali utilizzando dati provenienti da documenti PDF caricati e processati.

## Caratteristiche

- **Unione PDF**: Unisce multipli file PDF in un singolo documento per facilitare la ricerca.
- **Creazione di Embeddings**: Utilizza modelli di embedding avanzati per trasformare il testo in rappresentazioni vettoriali.
- **Vector Store**: Crea e gestisce un vector store per una rapida ricerca e recupero di informazioni.
- **Chatbot Interattivo**: Fornisce un'interfaccia utente basata su Streamlit per interagire con il chatbot.

## Componenti Principali

### Script di Preparazione (`preparazione.py`)
- Unisce i PDF presenti in una cartella specificata.
- Legge e divide il documento PDF in chunk.
- Crea gli embeddings e salva il vector store per l'utilizzo successivo.

### Script del Chatbot (`chatbot.py`)
- Utilizza il vector store creato per rispondere alle domande degli utenti.
- Fornisce un'interfaccia utente tramite Streamlit per interagire con il chatbot.

## Requisiti

- Python 3.8 o superiore
- Installare le dipendenze con:
  ```sh
  pip install -r requirements.txt
