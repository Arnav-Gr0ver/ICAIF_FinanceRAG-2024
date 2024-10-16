from sentence_transformers import CrossEncoder
import logging
import pandas as pd
import os

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
    "TATQA": TATQA(),
}

output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

file_paths = []

for task_name, task_instance in tasks.items():
    logging.info(f"Processing task: {task_name}")
    
    encoder_model = SentenceTransformerEncoder(
        model_name_or_path='intfloat/e5-large-v2',
        query_prompt='query: ',
        doc_prompt='passage: ',
    )
    
    retrieval_model = DenseRetrieval(
        model=encoder_model
    )
    
    retrieval_result = task_instance.retrieve(
        retriever=retrieval_model
    )
    
    reranker = CrossEncoderReranker(
        model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    )
    
    reranking_result = task_instance.rerank(
        reranker=reranker,
        results=retrieval_result,
        top_k=100,
        batch_size=32
    )
    
    result_df = pd.DataFrame(reranking_result, columns=['query_id', 'corpus_id'])

    file_path = f"{output_dir}/{task_name}_results.csv"
    result_df.to_csv(file_path, index=False)
    file_paths.append(file_path)

    logging.info(f"Results for task {task_name} saved to {file_path}")

all_results = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

consolidated_file_path = f"{output_dir}/consolidated_results.csv"
all_results.to_csv(consolidated_file_path, index=False)

logging.info(f"All results have been consolidated and saved to {consolidated_file_path}")

print(f"All results have been consolidated and saved to {consolidated_file_path}")