from ragatouille import RAGPretrainedModel
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

tasks = ["ConvFinQA", "FinDER", "FinQA", "FinQABench", "FinanceBench", "MultiHiertt", "TATQA"]
results_data = []

def generate_query_variations(query, num_variations=3):
    paraphrases = paraphrase_model.encode([query] * num_variations, convert_to_tensor=True)
    paraphrased_queries = [util.paraphrase_generation(query) for query in paraphrases]
    return paraphrased_queries

for task in tasks:
    corpus_dataset = load_dataset("Linq-AI-Research/FinanceRAG", task, split="corpus")
    query_dataset = load_dataset("Linq-AI-Research/FinanceRAG", task, split="queries")
    corpus_df = pd.DataFrame(corpus_dataset)
    unique_corpus_df = corpus_df.drop_duplicates(subset="_id")
    index = RAG.index(
        collection=unique_corpus_df["text"].tolist(),
        document_ids=unique_corpus_df["_id"].tolist()
    )

    for i, query_data in enumerate(query_dataset):
        query_id = query_data["_id"]
        query = query_data["text"]
        query_variations = generate_query_variations(query, num_variations=3)
        doc_scores = {}

        for q_variation in query_variations:
            results = RAG.search(q_variation)
            for result in results:
                doc_id = result["document_id"]
                score = result["score"]
                
                if doc_id in doc_scores:
                    doc_scores[doc_id].append(score)
                else:
                    doc_scores[doc_id] = [score]

        ranked_results = sorted(
            [(doc_id, sum(scores) / len(scores)) for doc_id, scores in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for result_id, avg_score in ranked_results:
            results_data.append({
                "query_id": query_id,
                "result_id": result_id,
                "avg_score": avg_score
            })

        print(f"Processed query {i} for task {task}")

results_df = pd.DataFrame(results_data)
results_df.to_csv("results/rag_multiquery_top10.csv", index=False, encoding="utf-8")