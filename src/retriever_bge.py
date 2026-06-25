import math
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

EMBED_MODEL = 'BAAI/bge-base-en-v1.5'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
INDEX_PATH = 'index/faiss_index'
INITIAL_K = 30
FINAL_K = 5
MAX_QNA = 2

BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

def _sigmoid(x: float) -> float:
    return round(1/(1+math.exp(-x)),3)

class TeleRAGRetriever:
    def __init__(self):
        print('Loading Embedded Model...')
        self.embeddings = HuggingFaceEmbeddings(
            model_name = EMBED_MODEL, 
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True},
        )

        print('Loading FAISS index...')
        self.vectorstore = FAISS.load_local(INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

        print('Building BM25 keyword index...')
        self._build_bm25()

        print('Loading re-ranker (downloads ~80MB on first run)...')
        self.reranker = CrossEncoder(RERANK_MODEL)
        print('Retriever ready')

    def _build_bm25(self):
        """Build BM25 over all documents stored in FAISS docstore."""
        docstore       = self.vectorstore.docstore._dict
        self.bm25_docs = list(docstore.values())
        tokenized      = [doc.page_content.lower().split()
                          for doc in self.bm25_docs]
        self.bm25      = BM25Okapi(tokenized)
        print(f'  BM25 index built over {len(self.bm25_docs)} documents')

    def retrieve(self, query: str) -> list:
 
        # ── Stage 1a: FAISS semantic search (BGE-Large) ───────
        # CHANGE 1: prepend BGE query prefix for best results
        bge_query     = BGE_QUERY_PREFIX + query
        semantic_hits = self.vectorstore.similarity_search(bge_query, k=INITIAL_K)
 
        # ── Stage 1b: BM25 keyword search ─────────────────────
        # CHANGE 2: catches exact telecom acronyms FAISS misses
        # e.g. "E2SM-KPM", "PDCCH", "gNB" match exactly here
        tokenized_query = query.lower().split()
        bm25_scores     = self.bm25.get_scores(tokenized_query)
        top_bm25_idx    = np.argsort(bm25_scores)[::-1][:INITIAL_K]
        bm25_hits       = [self.bm25_docs[i] for i in top_bm25_idx]
 
        # ── Stage 1c: merge and deduplicate ───────────────────
        seen, merged = set(), []
        for doc in semantic_hits + bm25_hits:
            key = doc.page_content[:80]   # deduplicate by first 80 chars
            if key not in seen:
                seen.add(key)
                merged.append(doc)
 
        # ── Stage 2: CrossEncoder reranking ───────────────────
        pairs       = [[query, doc.page_content] for doc in merged]
        raw_scores  = self.reranker.predict(pairs)
        norm_scores = [_sigmoid(s) for s in raw_scores]
 
        ranked = sorted(
            zip(merged, norm_scores),
            key=lambda x: x[1],
            reverse=True,
        )
 
        # ── Stage 3: source diversity filter ──────────────────
        final, qna_count = [], 0
        for doc, score in ranked:
            is_qna = doc.metadata.get('type') == 'qna'
            if is_qna:
                if qna_count < MAX_QNA:
                    final.append((doc, score))
                    qna_count += 1
            else:
                final.append((doc, score))
            if len(final) >= FINAL_K:
                break
 
        # ── Stage 4: return parent chunks to LLM ──────────────
        # CHANGE 3: return the large parent text for richer LLM context
        # The small child chunk was used for precise retrieval above.
        # Now swap back to the parent for the actual answer generation.
        enriched = []
        for doc, score in final:
            parent_text  = doc.metadata.get('parent_text', doc.page_content)
            enriched_doc = Document(
                page_content = parent_text,
                metadata     = doc.metadata,
            )
            enriched.append((enriched_doc, score))
 
        return enriched

if __name__ == '__main__':
    r       = TeleRAGRetriever()
    results = r.retrieve('What is the purpose of PDCCH in 5G NR?')
    print(f'\nTop {len(results)} results:')
    for i, (doc, score) in enumerate(results):
        print(f'[{i+1}] Score: {score:.3f} | '
              f'{doc.metadata.get("type")} | '
              f'{doc.metadata.get("source")} p.{doc.metadata.get("page")}')
        print(f'    {doc.page_content[:150]}...')