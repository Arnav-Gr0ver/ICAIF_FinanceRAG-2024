from ragatouille import RAGPretrainedModel
from datasets import load_dataset
import pandas as pd
import transformers
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

PIPELINE = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", max_new_tokens=50, truncation=True)
RAG = RAGPretrainedModel.from_pretrained("SesameStreet/FinColBERT")
TASKS = ["ConvFinQA", "FinDER", "FinQA", "FinQABench", "FinanceBench", "MultiHiertt", "TATQA"]

results_data = []

for task in TASKS:
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

        PROMPT = f"""
        Answer the following query:
        {query}
        think through the rationale before answering
        """

        CoT_query = query + " " + PIPELINE(PROMPT)[0]['generated_text'].split("$$$")[1]
        results = RAG.search(CoT_query)

        for result in results:
            results_data.append({
              "query_id": query_id, 
              "result_id": result["document_id"],
            })
        print(f"{task} : {i}")

results_df = pd.DataFrame(results_data)
results_df.to_csv("data/results.csv", index=False, encoding="utf-8")