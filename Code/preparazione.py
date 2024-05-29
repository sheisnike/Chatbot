import os
import torch
import gc
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfMerger

'''merge_pdfs_from_folder è utilizzata 
per unire tutti i file PDF presenti in una specifica 
cartella (e sottocartelle) in un unico file PDF di output'''

def merge_pdfs_from_folder(folder_path, output_path):
    # Crea un oggetto PdfMerger, che sarà usato per unire i file PDF.
    merger = PdfMerger()
    # Usa os.walk per scorrere ricorsivamente tutte le cartelle e sottocartelle del percorso specificato.
    for root, dirs, files in os.walk(folder_path):
        # Scorre tutti i file trovati nella cartella corrente.
        for file in files:
            # Controlla se il file ha l'estensione '.pdf' (non case-sensitive).
            if file.lower().endswith('.pdf'):
                # Costruisce il percorso completo del file.
                file_path = os.path.join(root, file)
                # Stampa il percorso del file che sarà aggiunto.
                print(f'Adding {file_path}')
                # Aggiunge il file PDF al PdfMerger.
                merger.append(file_path)
    # Scrive tutti i PDF uniti nel file di output specificato.
    merger.write(output_path)
    # Chiude il PdfMerger per rilasciare le risorse.
    merger.close()
    # Stampa un messaggio indicando il completamento dell'operazione.
    print(f'Merged PDF saved to {output_path}')

'''La funzione clear_memory è utilizzata per liberare 
la memoria sia dalla GPU (utilizzando PyTorch) che 
dalla RAM (utilizzando il modulo gc di Python).'''

def clear_memory():
    # Libera la cache della memoria della GPU utilizzando PyTorch.
    torch.cuda.empty_cache()
    # Effettua la garbage collection manuale per liberare la memoria nella RAM.
    gc.collect()

'''La funzione read_pdf legge un file PDF e estrae il testo 
contenuto in tutte le sue pagine, restituendo il testo completo 
come una singola stringa.'''

def read_pdf(file):
    # Inizializza una stringa vuota per accumulare il testo estratto.
    document = ""
    
    # Crea un oggetto PdfReader per leggere il file PDF specificato.
    reader = PdfReader(file)
    
    # Itera attraverso tutte le pagine del PDF.
    for page in reader.pages:
        # Estrae il testo dalla pagina corrente e lo aggiunge alla stringa document.
        document += page.extract_text()
    
    # Restituisce il testo completo estratto dal PDF.
    return document

'''La funzione split_doc divide un documento di testo in più parti 
(o chunk) utilizzando un metodo di splitting ricorsivo, 
basato su caratteri. Questa tecnica è utile per processare
 testi lunghi in blocchi più piccoli, mantenendo un certo livello di sovrapposizione 
 tra i chunk per garantire la coerenza del contesto'''

def split_doc(document, chunk_size, chunk_overlap):
    # Crea un oggetto RecursiveCharacterTextSplitter con la dimensione del chunk e la sovrapposizione specificate.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    # Divide il documento di testo nei chunk specificati utilizzando il metodo split_text.
    split = splitter.split_text(document)  
    # Crea documenti dai chunk suddivisi per ulteriore processamento.
    split = splitter.create_documents(split) 
    # Restituisce i documenti suddivisi.
    return split

'''La funzione embedding_storing crea e gestisce un
vector store utilizzando embedding generati da un modello detto sentence-transformers/all-MiniLM-L6-v2.
A seconda del parametro create_new_vs, la funzione può creare un nuovo vector store o aggiornare uno esistente.'''

def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    # Inizializza un modello di embedding utilizzando la libreria HuggingFace e specificando di usare la CPU.
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",                                           model_kwargs={'device': 'cpu'})  # Usa la CPU
    # Crea un database FAISS dai documenti suddivisi utilizzando gli embedding generati dal modello.
    db = FAISS.from_documents(split, instructor_embeddings)
    # Costruisce il percorso per salvare il vector store.
    vector_store_path = "vector_store/" + new_vs_name
    # Controlla se creare un nuovo vector store o aggiornare uno esistente.
    if create_new_vs:
        # Se deve creare un nuovo vector store, controlla se la directory esiste e se non esiste, la crea.
        if not os.path.exists(vector_store_path):
            os.makedirs(vector_store_path)
        # Salva il nuovo vector store nel percorso specificato.
        db.save_local(vector_store_path)
        print(f"Vector store saved to {vector_store_path}")
    else:
        # Se deve aggiornare un vector store esistente, lo carica dal percorso specificato.
        load_db = FAISS.load_local(
            vector_store_path,
            instructor_embeddings,
            allow_dangerous_deserialization=True #Quando viene impostato True, si sta esplicitamente dicendo 
                                                 #di consentire la deserializzazione di dati che potrebbero essere
                                                 #potenzialmente pericolosi è usato in situazioni in cui si è sicuri
                                                 #dell'origine e dell'integrità dei dati deserializzati
        )
        # Unisce il vector store caricato con il nuovo vector store creato dai documenti suddivisi.
        load_db.merge_from(db)
        # Salva il vector store unito nel percorso specificato.
        load_db.save_local(vector_store_path)
        print(f"Vector store merged and saved to {vector_store_path}")
    
    # Stampa un messaggio di conferma che il documento è stato salvato.
    print("Il documento è stato salvato.")

'''La funzione main esegue una serie di operazioni sui file PDF presenti in una cartella specifica. 
Prima unisce tutti i file PDF in un unico documento, poi cancella un vector store esistente (se presente),
legge il PDF unito, suddivide il testo in chunk e infine crea e salva un nuovo vector store basato sugli
embedding dei chunk di testo.'''

def main():
    # Specifica il percorso della cartella contenente i file PDF da unire.
    folder_path = '/Users/nikedigiacomo/Desktop/Parlamento'
    # Specifica il percorso e il nome del file PDF di output.
    output_path = '/Users/nikedigiacomo/Desktop/Prova/prova.pdf'
    # Specifica il nome del vector store da creare o aggiornare.
    vector_store_name = "nome_vector_store"

    # Unisce tutti i file PDF nella cartella specificata in un unico file PDF di output.
    merge_pdfs_from_folder(folder_path, output_path)

    # Costruisce il percorso del vector store basato sul nome specificato.
    vector_store_path = f"vector_store/" + vector_store_name
    # Se il vector store esiste già, lo cancella.
    if os.path.exists(vector_store_path):
        for file in os.listdir(vector_store_path):
            file_path = os.path.join(vector_store_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Cancella il file
        os.rmdir(vector_store_path)  # Rimuove la directory del vector store

    # Legge il contenuto del PDF unito in una stringa di testo.
    document = read_pdf(output_path)
    # Divide il documento in chunk di 500 caratteri con una sovrapposizione di 50 caratteri.
    split = split_doc(document, chunk_size=500, chunk_overlap=50)
    # Crea e salva un nuovo vector store basato sugli embedding dei chunk di testo.
    embedding_storing(split, create_new_vs=True, existing_vector_store="", new_vs_name=vector_store_name)

# Se questo script viene eseguito direttamente (non importato come modulo), chiama la funzione main.
if __name__ == "__main__":
    main()

