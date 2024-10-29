from ragatouille import RAGPretrainedModel
from datasets import load_dataset

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

finDER_dataset = load_dataset("Linq-AI-Research/FinanceRAG", "FinDER", split="corpus")

print(finDER_dataset)

# finDER_index = RAG.index(
#     index_name="finDER task corpus index"
#     collection=,
#     document_ids=
# )
