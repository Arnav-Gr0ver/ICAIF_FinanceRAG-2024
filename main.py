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

# List of tasks to be processed
tasks = {
    "FinDER": FinDER(),
    "ConvFinQA": ConvFinQA(),
    "FinQABench": FinQABench(),
    "FinanceBench": FinanceBench(),
    "MultiHiertt": MultiHiertt(),
    "TATQA": TATQA(),
}

# Directory to save individual task results
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# Initialize list to keep track of individual file paths
file_paths = []

# Loop through each task
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
    
    # Store results in a DataFrame
    result_df = pd.DataFrame(reranking_result, columns=['result'])
    result_df['task'] = task_name  # Add a column for the task name

    # Reorder columns to have 'task' first and then 'result'
    result_df = result_df[['task', 'result']]

    # Save individual task results to a separate CSV file
    file_path = f"{output_dir}/{task_name}_results.csv"
    result_df.to_csv(file_path, index=False)
    file_paths.append(file_path)  # Keep track of the file path

    logging.info(f"Results for task {task_name} saved to {file_path}")

# Concatenate all individual task CSV files vertically into one consolidated file
all_results = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# Save consolidated results to a single CSV file with a proper header
consolidated_file_path = f"{output_dir}/consolidated_results.csv"
all_results.to_csv(consolidated_file_path, index=False)

logging.info(f"All results have been consolidated and saved to {consolidated_file_path}")

print(f"All results have been consolidated and saved to {consolidated_file_path}")
