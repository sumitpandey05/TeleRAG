from langchain_ollama import OllamaLLM as Ollama
from src.retriever import TeleRAGRetriever

LLM_MODEL = 'llama3.1:8b'

class TeleRAGPipeline:
    def __init__(self):
        self.retriever = TeleRAGRetriever()
        self.llm = Ollama(model=LLM_MODEL,temperature=0)
        with open('prompts/system.txt', 'r') as f:
            self.system_prompt = f.read()
        print('Pipeline Ready')
    
    def _format_context(self, docs):
        parts = []
        for i, (doc,score) in enumerate(docs):
            src = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            parts.append(f'[context {i+1}] Source: {src}, Page: {page}\n{doc.page_content}')

        
        return '\n\n---\n\n'.join(parts)
    
    def query(self, question, verbose = False):
        retrieved = self.retriever.retrieve(question)
        if verbose:
            for doc, score in retrieved:
                print(f'[{score:.2f}] {doc.metadata["source"]}')
        context = self._format_context(retrieved)
        prompt = f'{self.system_prompt}\n\nCONTEXTPASSAGES: \n{context}\n\nQUESTION: {question}'
        answer = self.llm.invoke(prompt)
        source = [
            {'source': d.metadata.get('source', '?'), 'page': d.metadata.get('page', '?'), 'score': round(float(s), 3)}
            for d, s in retrieved
        ]

        return {'question': question, 'answer': answer, 'sources': source, 'retrieved': retrieved}
if __name__ == '__main__':
    p = TeleRAGPipeline()
    for q in ['What is the handover in 5G?', 'What does gNB stand for?']:
        print('\n' + '='*60)
        r = p.query(q)
        print(f'Q: {q}\nA: {r["answer"]}')