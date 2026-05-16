# src/evaluate.py
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision
from src.pipeline import TeleRAGPipeline

TELEQNA_PATH  = 'data/teleqna/TeleQnA.json'
NUM_QUESTIONS = 10        # keep at 10 while testing; raise to 50 for final run
OLLAMA_MODEL  = 'llama3.1:8b'

# ── Load held-out test questions ──────────────────────────────
def load_test_questions(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_items  = list(data.values())
    test_start = int(len(all_items) * 0.8)
    subset     = all_items[test_start : test_start + n]
    questions, ground_truths = [], []
    for item in subset:
        q          = item.get('question', '').strip()
        answer_raw = item.get('answer', '')
        answer_text = answer_raw.split(':', 1)[1].strip() if ':' in answer_raw else answer_raw.strip()
        if q and answer_text:
            questions.append(q)
            ground_truths.append(answer_text)
    return questions, ground_truths

# ── Main ──────────────────────────────────────────────────────
def run_eval():
    print('Setting up Ollama for RAGAs (no OpenAI needed)...')

    # Longer timeout stops Ollama from timing out on slow CPU
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model=OLLAMA_MODEL, temperature=0, timeout=180)
    )

    # 3 metrics only — AnswerRelevancy removed (slowest, needs embeddings)
    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print('Loading TeleRAG pipeline...')
    pipeline = TeleRAGPipeline()

    print(f'Loading {NUM_QUESTIONS} test questions...')
    questions, ground_truths = load_test_questions(TELEQNA_PATH, NUM_QUESTIONS)

    print(f'\nRunning {len(questions)} questions through pipeline...')
    answers, contexts = [], []
    for i, q in enumerate(questions):
        print(f'  {i+1}/{len(questions)}: {q[:60]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        contexts.append([doc.page_content for doc, _ in r['retrieved']])

    ds = Dataset.from_dict({
        'question':     questions,
        'answer':       answers,
        'contexts':     contexts,
        'ground_truth': ground_truths
    })

    print('\nScoring with RAGAs — this takes a few minutes per question...')
    results = evaluate(dataset=ds, metrics=metrics)

    # ragas 0.2.x returns lists — take the mean of each
    scores = results.to_pandas()
    faith     = scores['faithfulness'].mean()
    recall    = scores['context_recall'].mean()
    precision = scores['context_precision'].mean()

    print('\n========== TELERAG EVALUATION RESULTS ==========')
    print(f'  Faithfulness:      {faith:.3f}   (target > 0.90)')
    print(f'  Context Recall:    {recall:.3f}   (target > 0.85)')
    print(f'  Context Precision: {precision:.3f}   (target > 0.75)')
    print('=================================================')

    scores.to_csv('evaluation_results.csv', index=False)
    print('Saved to evaluation_results.csv')

if __name__ == '__main__':
    run_eval()