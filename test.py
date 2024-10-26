import pandas as pd
import logging
from datasets import load_dataset
from ragatouille import RAGPretrainedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load the FinDER dataset
logging.info("Loading the FinDER dataset.")
finDER_corpus = load_dataset("Linq-AI-Research/FinanceRAG", "FinDER", split="corpus")
finDER_queries = load_dataset("Linq-AI-Research/FinanceRAG", "FinDER", split="queries")

# Extract documents and IDs for indexing
logging.info("Extracting documents and IDs from the corpus.")
documents = [doc['text'] for doc in finDER_corpus]
document_ids = [doc['_id'] for doc in finDER_corpus]
document_metadatas = [{"title": doc['title']} for doc in finDER_corpus]

# Step 2: Index the documents
logging.info("Indexing documents with RAGPretrainedModel.")
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
index_path = RAG.index(
    index_name="finDER_index",
    collection=documents,
    document_ids=document_ids,
    document_metadatas=document_metadatas,
)
logging.info("Indexing completed successfully. Index path: %s", index_path)

# Step 3: Perform retrieval for each query and collect top 10 results
logging.info("Starting retrieval process for each query.")
results = []
for query in finDER_queries:
    query_text = query['text']
    query_id = query['_id']
    logging.info("Retrieving top 10 documents for query ID: %s", query_id)
    
    # Retrieve top 10 documents
    retrieval_results = RAG.search(query_text, top_k=10)
    
    # Collect results with query_id and corresponding corpus_ids
    for result in retrieval_results:
        results.append({
            "query_id": query_id,
            "corpus_id": result['document_id']
        })
    logging.info("Retrieved and stored top 10 results for query ID: %s", query_id)

# Step 4: Save results to a CSV
logging.info("Saving results to CSV.")
results_df = pd.DataFrame(results)
results_df.to_csv("finDER_retrieval_results.csv", index=False)
logging.info("Top 10 retrieval results per query have been saved to finDER_retrieval_results.csv.")
