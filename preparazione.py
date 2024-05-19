import os
import torch
import gc
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfMerger

def merge_pdfs_from_folder(folder_path, output_path):
    merger = PdfMerger()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                print(f'Adding {file_path}')
                merger.append(file_path)
    merger.write(output_path)
    merger.close()
    print(f'Merged PDF saved to {output_path}')

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split

def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                  model_kwargs={'device': 'cpu'})  # Usa la CPU

    db = FAISS.from_documents(split, instructor_embeddings)

    vector_store_path = "vector_store/" + new_vs_name
    if create_new_vs:
        if not os.path.exists(vector_store_path):
            os.makedirs(vector_store_path)
        db.save_local(vector_store_path)
        print(f"Vector store saved to {vector_store_path}")
    else:
        load_db = FAISS.load_local(
            vector_store_path,
            instructor_embeddings,
            allow_dangerous_deserialization=True
        )
        load_db.merge_from(db)
        load_db.save_local(vector_store_path)
        print(f"Vector store merged and saved to {vector_store_path}")
    
    print("Il documento è stato salvato.")

def main():
    folder_path = '/Users/nikedigiacomo/Desktop/Parlamento'
    output_path = '/Users/nikedigiacomo/Desktop/Prova/prova.pdf'
    vector_store_name = "nome_vector_store"

    merge_pdfs_from_folder(folder_path, output_path)

    vector_store_path = f"vector_store/" + vector_store_name
    if os.path.exists(vector_store_path):
        for file in os.listdir(vector_store_path):
            file_path = os.path.join(vector_store_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(vector_store_path)

    document = read_pdf(output_path)
    split = split_doc(document, chunk_size=500, chunk_overlap=50)
    embedding_storing(split, create_new_vs=True, existing_vector_store="", new_vs_name=vector_store_name)

if __name__ == "__main__":
    main()
