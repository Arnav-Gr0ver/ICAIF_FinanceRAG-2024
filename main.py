from sentence_transformers import CrossEncoder
import logging
import os
import pandas as pd

from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.tasks import (
    ConvFinQA,
    FinDER,
    FinQABench,
    FinanceBench,
    MultiHiertt,
    TATQA,
)

logging.basicConfig(level=logging.INFO)

tasks = {
    "FinDER": FinDER(),
    "ConvFinQA": ConvFinQA(),
    "FinQABench": FinQABench(),
    "FinanceBench": FinanceBench(),
    "MultiHiertt": MultiHiertt(),
    "TATQA": TATQA()
}

encoder_model = SentenceTransformerEncoder(
    model_name_or_path='intfloat/e5-large-v2',
    query_prompt='query: ',
    doc_prompt='passage: ',
)

retrieval_model = DenseRetrieval(
    model=encoder_model
)

reranker = CrossEncoderReranker(
    model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
)

results_list = []

for task_name, task in tasks.items():
    logging.info(f"Processing task: {task_name}")

    retrieval_result = task.retrieve(
        retriever=retrieval_model
    )

    reranking_result = task.rerank(
        reranker=reranker,
        results=retrieval_result,
        top_k=100,
        batch_size=32
    )

    output_dir = f'./results/{task_name}'
    task.save_results(output_dir=output_dir)

    task_results_file = os.path.join(output_dir, 'results.csv')
    if os.path.exists(task_results_file):
        task_results = pd.read_csv(task_results_file)
        task_results['Task'] = task_name
        results_list.append(task_results)

if results_list:
    all_results = pd.concat(results_list, ignore_index=True)
    all_results.to_csv('./results/combined_results.csv', index=False)