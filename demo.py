# demo.py
from src.pipeline import TeleRAGPipeline
from src.anomaly import AnomalyDetector
 
def run_demo():
    print('Loading TeleRAG...')
    p = TeleRAGPipeline()
    a = AnomalyDetector(p)
 
    demo_cases = [
        # (type, label, query)
        ('qna',     '3GPP QnA',           'What is the purpose of PDCCH in 5G NR?'),
        ('qna',     'O-RAN Architecture', 'What is the role of the RIC in O-RAN architecture?'),
        ('rca',     'Root Cause Analysis','UE is experiencing repeated handover failures in 5G NR'),
        ('anomaly', 'Anomaly Detection',  'CQI dropped from 12 to 3, BLER spiked to 40%, RLF reported on cell 47'),
    ]
 
    for case_type, label, query in demo_cases:
        print('\n' + '='*65)
        print(f'USE CASE: {label}')
        print(f'QUERY:    {query}')
        print('='*65)
 
        if case_type == 'qna':
            r = p.query(query)
            print(f'ANSWER:\n{r["answer"]}')
            print(f'\nRESPONSE TIME: {r["total_time_s"]}s')
            print('SOURCES:')
            for s in r['sources']:
                print(f'  [{s["type"]}] {s["source"]} — page {s["page"]} (score {s["score"]})')
 
        elif case_type == 'rca':
            r = p.rca_query(query)
            print(f'ANALYSIS:\n{r["answer"]}')
            print('SOURCES:')
            for s in r['sources']:
                print(f'  [{s["type"]}] {s["source"]} — page {s["page"]} (score {s["score"]})')
 
        elif case_type == 'anomaly':
            r = a.analyze(query)
            print(f'IS ANOMALY:       {r["is_anomaly"]}')
            print(f'FLAGGED KEYWORDS: {r["flagged_keywords"]}')
            print(f'ANALYSIS:\n{r["rag_analysis"]}')
            print('SOURCES:')
            for s in r['sources']:
                print(f'  [{s["type"]}] {s["source"]} — page {s["page"]} (score {s["score"]})')
 
if __name__ == '__main__':
    run_demo()