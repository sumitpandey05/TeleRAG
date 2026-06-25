# src/ingest.py — MiniLM version (revert for screenshots)
import os, json
import fitz
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PDF_FOLDER   = 'data/pdfs'
ORAN_FOLDER  = 'data/oransc'
INDEX_PATH   = 'index/faiss_index'
TELEQNA_PATH = 'data/teleqna/TeleQnA.json'

EMBED_MODEL   = 'sentence-transformers/all-MiniLM-L6-v2'   # reverted from BGE
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200


def extract_pdf(path):
    doc   = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text('text')
        if len(text.strip()) > 50:
            pages.append({
                'text':   text,
                'page':   i + 1,
                'source': os.path.basename(path),
            })
    doc.close()
    return pages


def chunk_pages(pages, doc_type='3GPP'):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        separators      = ['\n\n', '\n', '. ', ' ', ''],
        add_start_index = True,
    )
    chunks = []
    for p in pages:
        for split in splitter.split_text(p['text']):
            chunks.append(Document(
                page_content = split,
                metadata     = {
                    'source': p['source'],
                    'page':   p['page'],
                    'type':   doc_type,
                }
            ))
    return chunks


def load_teleqna(path):
    if not os.path.exists(path):
        print(f'Warning: TeleQnA not found at {path}. Skipping.')
        return []

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for key, item in data.items():
        q           = item.get('question', '').strip()
        answer_raw  = item.get('answer', '')
        explanation = item.get('explanation', '')
        category    = item.get('category', '')

        answer_text = (
            answer_raw.split(':', 1)[1].strip()
            if ':' in answer_raw else answer_raw.strip()
        )

        text = (
            f"Question: {q}\n"
            f"Answer: {answer_text}\n"
            f"Explanation: {explanation}\n"
            f"Category: {category}"
        )

        docs.append(Document(
            page_content = text,
            metadata     = {
                'source':      'TeleQnA',
                'page':        0,
                'type':        'qna',
                'question_id': key,
            }
        ))

    print(f'Loaded {len(docs)} TeleQnA Q&A pairs')
    return docs


def main():
    os.makedirs('index', exist_ok=True)
    all_docs = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    print(f'\nProcessing {len(pdf_files)} 3GPP PDFs...')
    for pdf in tqdm(pdf_files, desc='3GPP PDFs'):
        pages  = extract_pdf(os.path.join(PDF_FOLDER, pdf))
        chunks = chunk_pages(pages, doc_type='3GPP')
        all_docs.extend(chunks)
        print(f'  {pdf}: {len(pages)} pages → {len(chunks)} chunks')

    if os.path.exists(ORAN_FOLDER):
        oran_files = [f for f in os.listdir(ORAN_FOLDER) if f.endswith('.pdf')]
        print(f'\nProcessing {len(oran_files)} O-RAN specs...')
        for pdf in tqdm(oran_files, desc='O-RAN PDFs'):
            pages  = extract_pdf(os.path.join(ORAN_FOLDER, pdf))
            chunks = chunk_pages(pages, doc_type='O-RAN')
            all_docs.extend(chunks)
            print(f'  {pdf}: {len(pages)} pages → {len(chunks)} chunks')
    else:
        print(f'\nWarning: {ORAN_FOLDER} not found. Skipping.')

    all_docs.extend(load_teleqna(TELEQNA_PATH))
    print(f'\nTotal documents to index: {len(all_docs)}')

    print('Loading MiniLM embedding model...')
    embeddings = HuggingFaceEmbeddings(
        model_name   = EMBED_MODEL,
        model_kwargs = {'device': 'cpu'},
    )

    print('Building FAISS index...')
    vectorstore = None
    for i in tqdm(range(0, len(all_docs), 500), desc='Batches'):
        batch = all_docs[i : i + 500]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local(INDEX_PATH)
    print(f'\nIndex saved to {INDEX_PATH}')
    print(f'Total vectors in index: {vectorstore.index.ntotal}')


if __name__ == '__main__':
    main()