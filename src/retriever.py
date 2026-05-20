import math
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
INDEX_PATH = 'index/faiss_index'
INITIAL_K = 30
FINAL_K = 5
MAX_QNA = 2

def _sigmoid(x: float) -> float:
    return round(1/(1+math.exp(-x)),3)

class TeleRAGRetriever:
    def __init__(self):
        print('Loading Embedded Model...')
        self.embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL, model_kwargs = {'device': 'cpu'})

        print('Loading FAISS index...')
        self.vectorstore = FAISS.load_local(INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

        print('Loading re-ranker (downloads ~80MB on first run)...')
        self.reranker = CrossEncoder(RERANK_MODEL)
        print('Retriever ready')

    def retrieve(self,query:str)->list:
        #Stage 1: Broad FAISS search
        candidates = self.vectorstore.similarity_search_with_score(query,k=INITIAL_K)
        #Stage 2: RE-Rank with the cross encoder

        pairs = [[query,doc.page_content] for doc, _ in candidates]
        scores = self.reranker.predict(pairs)

        norm_scores = [_sigmoid(s) for s in scores]

        ranked = sorted(zip(candidates, norm_scores), key = lambda x: x[1], reverse = True)

        final = []
        qna_count = 0

        for(doc,_), norm_score in ranked:
            is_qna = doc.metadata.get('type') == 'qna'

            if is_qna:
                if qna_count < MAX_QNA:
                    final.append((doc, norm_score))
                    qna_count+=1
            
            else:
                final.append((doc,norm_score))


            if len(final) >= FINAL_K:
                break

        return final

if __name__ == '__main__':
    r = TeleRAGRetriever()
    results = r.retrieve('What is the purpose of PDCCH in 5G NR?')
    print(f'\nTop {len(results)} results:')
    for i, (doc, score) in enumerate(results):
        print(f'[{i+1}] Score: {score:.3f} | Type: {doc.metadata["type"]} | {doc.metadata["source"]}')
        print(f'    {doc.page_content[:150]}...')