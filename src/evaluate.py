import sys, os, json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from src.pipeline import TeleRAGPipeline

TELEQNA_PATH  = 'data/teleqna/TeleQnA.json'
NUM_QUESTIONS = 50  
OLLAMA_MODEL  = 'llama3.1:8b'

def load_test_questions(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_items = list(data.values())
    test_start = int(len(all_items) * 0.8)   # held-out last 20%
    subset = all_items[test_start : test_start + n]
    questions, ground_truths = [], []
    for item in subset:
        q          = item.get('question', '').strip()
        answer_raw = item.get('answer', '')
        answer_text = answer_raw.split(':', 1)[1].strip() if ':' in answer_raw else answer_raw.strip()
        if q and answer_text:
            questions.append(q)
            ground_truths.append(answer_text)
    return questions, ground_truths

def run_eval():
    print('Configuring Ollama as judge LLM for RAGAs...')

    # Point RAGAs at your local Ollama — fully private, free
    ragas_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0))
    ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=OLLAMA_MODEL))

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print('Loading TeleRAG pipeline...')
    pipeline = TeleRAGPipeline()

    print(f'Loading {NUM_QUESTIONS} held-out test questions...')
    questions, ground_truths = load_test_questions(TELEQNA_PATH, NUM_QUESTIONS)

    print(f'\nEvaluating on {len(questions)} questions...')
    answers, contexts = [], []
    for i, q in enumerate(questions):
        print(f'  {i+1}/{len(questions)}: {q[:60]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        contexts.append([doc.page_content for doc, _ in r['retrieved']])

    # Build the dataset RAGAs expects
    ds = Dataset.from_dict({
        'question':    questions,
        'answer':      answers,
        'contexts':    contexts,
        'ground_truth': ground_truths
    })

    print('\nRunning RAGAs scoring with Ollama — this takes a few minutes...')
    results = evaluate(dataset=ds, metrics=metrics)

    print('\n========== TELERAG EVALUATION RESULTS ==========')
    print(f'  Faithfulness:      {results["faithfulness"]:.3f}   (target > 0.90)')
    print(f'  Answer Relevancy:  {results["answer_relevancy"]:.3f}   (target > 0.85)')
    print(f'  Context Recall:    {results["context_recall"]:.3f}   (target > 0.85)')
    print(f'  Context Precision: {results["context_precision"]:.3f}   (target > 0.75)')
    print('=================================================')

    results.to_pandas().to_csv('evaluation_results.csv', index=False)
    print('Detailed results saved to evaluation_results.csv')

if __name__ == '__main__':
    run_eval()