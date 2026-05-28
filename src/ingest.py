import os #used for working with folders/files ,joining file paths, listing pdfs in a directory
import json #used to read the teleqna json data set
import fitz #PyMuPDF library, used to open and read files page by page
from tqdm import tqdm #Creates progress bars while processing files and batches
from langchain_text_splitters import RecursiveCharacterTextSplitter #splits large texts into smaller chunks
from langchain_community.vectorstores import FAISS #FAISS = facebook AI similarity search, Used to store vectors embeddings for semantic search
from langchain_huggingface import HuggingFaceEmbeddings #Converts text into embeddings/vector using HuggingFace models
from langchain_core.documents import Document #Standard Langchain Document object, it stores the page content and the metadata 

PDF_FOLDER = 'data/pdfs' #Folder with the telecom pdfs
ORAN_FOLDER = 'data/oransc' #Folder with ORAN specifications
INDEX_PATH = 'index/faiss_index' #Path where the FAISS index will be saved after processing the documents
TELEQNA_PATH = 'data/teleqna/TeleQnA.json' #Path to the teleqna dataset

EMBED_MODEL = 'BAAI/bge-large-en-v1.5' #Embedding model name that converts text into vectors

CHILD_CHUNK_SIZE = 512 #Each chunk will contain a maximum of 2000 characters
CHILD_OVERLAP = 50 #Adjacent chunks overlap by 50 chars, this helps preserve the context in the text
PARENT_CHUNK_SIZE = 2048
PARENT_OVERLAP = 200

#Function to extract text from a pdf file, it uses the PyMuPDF library to read the file page by page and extract the text content. It returns a list of dictionaries, where each dictionary contains the text, page number, and source filename for each page that has more than 50 characters of text.
def extract_pdf(path):

    doc = fitz.open(path)

    pages = []

    for i, page in enumerate(doc):

        text = page.get_text('text')

        if len(text.strip()) > 50:
            pages.append({'text':text, 
                          'page':i+1, 
                          'source':os.path.basename(path),
            })

    doc.close()

    return pages

def chunk_pages(pages,doc_type = '3GPP'):

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size = PARENT_CHUNK_SIZE, 
        chunk_overlap = PARENT_OVERLAP, 
        separators = ['\n\n', '\n', '. ', ' ', ''],
        add_start_index = True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHILD_CHUNK_SIZE,
        chunk_overlap   = CHILD_OVERLAP,
        separators      = ['\n\n', '\n', '. ', ' ', ''],
        add_start_index = True,
    )

    child_docs = []
    

    for p in pages:
        parent_chunks = parent_splitter.split_text(p['text'])
        for parent_text in parent_chunks:
            child_chunks = child_splitter.split_text(parent_text)
            for child_text in child_chunks:
                child_docs.append(Document(
                    page_content = child_text,       # small — for retrieval
                    metadata     = {
                        'source':      p['source'],
                        'page':        p['page'],
                        'type':        doc_type,
                        'parent_text': parent_text,  # large — for LLM context
                    }
                ))
    return child_docs

def load_teleqna(path):

    if not os.path.exists(path):
        print(f'Warning: TeleQnA dataset not found at {path}.Skipping')
        return []
    
    with open(path,'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []

    for key,item in data.items():
        q          = item.get('question', '').strip()
        answer_raw = item.get('answer', '')        # format: "option 3: min(nt, nr)"
        explanation= item.get('explanation', '')
        category   = item.get('category', '')

        if ':' in answer_raw:
            answer_text = answer_raw.split(':', 1)[1].strip()
        else:
            answer_text = answer_raw.strip()

        text = (
            f"Question: {q}\n"
            f"Answer: {answer_text}\n"
            f"Explanation: {explanation}\n"
            f"Category: {category}"
        )

        docs.append(Document(
            page_content = text,
            metadata = {
                'source': 'TeleQnA',
                'page' : 0,
                'type' : 'qna',
                'question_id' : key,
            }
        ))
    print(f'Loaded {len(docs)} TeleQnA Q&A pairs')    
    return docs

def main():
    os.makedirs('index', exist_ok=True)
    all_docs = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    print(f'\nProcessing {len(pdf_files)} 3GPP PDFs...')
    for pdf in tqdm(pdf_files, desc='3GPP PDFS'):
        pages = extract_pdf(os.path.join(PDF_FOLDER, pdf))
        chunks = chunk_pages(pages)
        all_docs.extend(chunks)
        print(f'{pdf}: {len(pages)} pages -> {len(chunks)} chunks')
    
    if os.path.exists(ORAN_FOLDER):
        oran_files = [f for f in os.listdir(ORAN_FOLDER) if f.endswith('.pdf')]
        print(f'\nProcessing {len(oran_files)} O-RAN specs...')
        for pdf in tqdm(oran_files, desc='O-RAN PDFs'):
            pages  = extract_pdf(os.path.join(ORAN_FOLDER, pdf))
            chunks = chunk_pages(pages, doc_type='O-RAN')
            all_docs.extend(chunks)
            print(f'  {pdf}: {len(pages)} pages → {len(chunks)} child chunks')
    else:
        print(f'\nWarning: {ORAN_FOLDER} not found. Skipping.')
        
    teleqna_docs = load_teleqna(TELEQNA_PATH)
    all_docs.extend(teleqna_docs)

    print(f'Total documents to index: {len(all_docs)}')

    print('\nLoading BGE-Large embedding model (~1.3GB download on first run)...')
    
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True},
    )
    print('Building FAISS index')

    vectorstore = None
    for i in tqdm(
        range(0, len(all_docs), 500),desc='Batches'):
        batch = all_docs[i:i+500]

        if vectorstore is None:

            vectorstore = FAISS.from_documents(batch,embeddings)

        else:
            vectorstore.add_documents(batch)
    vectorstore.save_local(INDEX_PATH)
    print(f'Index saved to {INDEX_PATH}')
    print(f'Total vectors in index: {vectorstore.index.ntotal}')

if __name__ == '__main__':
    main()