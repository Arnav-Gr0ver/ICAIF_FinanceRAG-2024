from ragatouille import RAGPretrainedModel
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import torch

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
model = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

tasks = ["ConvFinQA", "FinDER", "FinQA", "FinQABench", "FinanceBench", "MultiHiertt", "TATQA"]
results_data = []

for task in tasks:
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

        prompt = f"Please write a scientific paper passage to answer the question: {query}"
        res = model(prompt)
        context = res[0]["generated_text"]
    
        query = f"""
            Context: {context}
            Question: {query}
        """

        print(query)

        results = RAG.search(query)

        for result in results:
            results_data.append({
              "query_id": query_id, 
              "result_id": result["document_id"],
            })
        print(i)

results_df = pd.DataFrame(results_data)
results_df.to_csv("results/rag_results.csv", index=False, encoding="utf-8")