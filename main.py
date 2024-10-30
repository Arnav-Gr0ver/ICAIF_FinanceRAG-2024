from ragatouille import RAGPretrainedModel
from transformers import pipeline
from datasets import load_dataset
import pandas as pd

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

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

        prompt = f"""Please write a scientific paper passage to answer the question
        Question: {query}
        Passage:"""

        messages = [
            {"role": "user", "content": prompt},
        ]
        pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
        pipe(messages)
    
        query = f"""
            Context: {pipe(messages)}
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