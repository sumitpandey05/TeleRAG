from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
INDEX_PATH = 'index/faiss_index'
INITIAL_K = 20
FINAL_K = 5

class TeleRAGRetriever:
    def __init__(self):
        print('Loading Embedded Model...')
        self.embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL, model_kwargs = {'device': 'cpu'})

        print('Loading FAISS index...')
        self.vectorstore = FAISS.load_local(INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

        print('Loading re-ranker (downloads ~80MB on first run)...')
        self.reranker = CrossEncoder(RERANK_MODEL)
        print('Retriever ready')

    def retrieve(self,query):
        #Stage 1: Broad FAISS search
        candidates = self.vectorstore.similarity_search_with_score(query,k=INITIAL_K)
        #Stage 2: RE-Rank with the cross encoder

        pairs = [[query,doc.page_content] for doc, _ in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key = lambda x: x[1], reverse = True)
        return [(doc,float(s)) for (doc, _), s in ranked[:FINAL_K]]

if __name__ == '__main__':
    r = TeleRAGRetriever()
    results = r.retrieve('What is handover in 5G NR?')
    print(f'Top {len(results)} results:')

    for i,(doc, score) in enumerate(results):
        print(f'[{i+1}] Score: {score:.3f} | {doc.metadata["source"]}')
        print(f'    {doc.page_content[:150]}...')