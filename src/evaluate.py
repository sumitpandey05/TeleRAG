import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from src.pipeline import TeleRAGpipeline

TELEQNA_PATH = 'data/teleqna/TeleQnA.json'
NUM_QUESTION = 50

def load_test_questions(path, n):
    with open(path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    test_start = int(len(data) * 0.8)
    subset = data[test_start: test_start+n]
    questions, ground_truths = [], []
    for item in subset:
        q = item.get('question', '')
        ans_key = item.get('answer', '')
        ans_text = item.get(ans_key, ans_key)
        if q and ans_text:
            questions.append(q)
            ground_truths.append(ans_text)
    
    return questions, ground_truths

def run_eval():
    pipeline = TeleRAGpipeline()
    questions, ground_truths = load_test_questions(TELEQNA_PATH, NUM_QUESTION)
    print(f'Evaluating on  {len(questions)} questions...')
    answers, contexts = [], []

    for i,q in enumerate(questions):
        print(f'{i+1}/{len(questions)}: {q[:50]}...')
        r = pipeline.query(q)
        answers.append(r['answer'])
        contexts.append([doc.page_content for doc, _ in r['retrieved']])

    ds = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts' : contexts,
        'ground_truths': ground_truths,
        
    })

    print('RAGAS metrics')
    results = evaluate(
        dataset = ds,
        metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_recall': context_recall,
            'context_precision': context_precision
        },
    )

    print("TELERAG EVALUATION RESULTS:")
    print(f'faithfulness: {results["faithfulness"]:.3f} (target>0.90)')
    print(f'answer_relevancy: {results["answer_relevancy"]:.3f} (target>0.85)')
    print(f'context_recall: {results["context_recall"]:.3f} (target>0.85)')
    print(f'context_precision: {results["context_precision"]:.3f} (target>0.75)')

    print('======================================================================')

    results.to_pandas().to_csv('evaluation_results.csv', index=False)
    print('Saved to evaluation_results.csv')

if __name__ == "__main__":
    run_eval()  