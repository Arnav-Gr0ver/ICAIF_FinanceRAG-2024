from ragatouille import RAGPretrainedModel
from datasets import load_dataset
import pandas as pd
import os

RAG = RAGPretrainedModel.from_pretrained("SesameStreet/FinColBERT")
TASKS = ["ConvFinQA", "FinDER", "FinQA", "FinQABench", "FinanceBench", "MultiHiertt", "TATQA"]

os.makedirs("data", exist_ok=True)

for task in TASKS:
    results_data = []
    corpus_dataset = load_dataset("Linq-AI-Research/FinanceRAG", task, split="corpus")
    query_dataset = load_dataset("Linq-AI-Research/FinanceRAG", task, split="queries")

    corpus_df = pd.DataFrame(corpus_dataset)
    unique_corpus_df = corpus_df.drop_duplicates(subset="_id")

    index = RAG.index(
        collection=unique_corpus_df["text"].tolist(),
        document_ids=unique_corpus_df["_id"].tolist()
    )

    for i in range(len(query_dataset)):
        query_id = query_dataset[i]["_id"]
        query = query_dataset[i]["text"]

        results = RAG.search(query)

        for result in results:
            results_data.append({
                "query_id": query_id,
                "corpus_id": result["document_id"],
            })

        print(f"{task} : Processed query {i+1}/{len(query_dataset)}")

    task_df = pd.DataFrame(results_data)
    task_df.to_csv(f"data/{task}_results.csv", index=False, encoding="utf-8")

all_results_df = pd.concat(
    [pd.read_csv(f"data/{task}_results.csv") for task in TASKS],
    ignore_index=True
)

all_results_df.to_csv("data/combined_results.csv", index=False, encoding="utf-8")