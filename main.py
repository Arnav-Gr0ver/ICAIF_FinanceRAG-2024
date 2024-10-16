from sentence_transformers import CrossEncoder
import logging
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

tasks = [
    ConvFinQA(),
    FinDER(),
    FinQABench(),
    FinanceBench(),
    MultiHiertt(),
    TATQA(),
]

results_df = pd.DataFrame(columns=["Task", "Query ID", "Document ID", "Score"])

for task in tasks:
    retrieval_result = task.retrieve(retriever=retrieval_model)
    reranking_result = task.rerank(
        reranker=reranker,
        results=retrieval_result,
        top_k=100,
        batch_size=32
    )
    
    for q_id, result in reranking_result.items():
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
        
        temp_df = pd.DataFrame([
            {
                "Task": type(task).__name__,
                "Query ID": q_id,
                "Document ID": doc_id,
                "Score": score
            } for doc_id, score in sorted_results
        ])
        
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

output_dir = './results'
results_df.to_csv(f"{output_dir}/combined_results.csv", index=False)

print(f"Results have been saved to {output_dir}/combined_results.csv")