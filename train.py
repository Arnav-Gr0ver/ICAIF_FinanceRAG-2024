from ragatouille import RAGTrainer
from datasets import load_dataset

qa_dataset = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")

trainer = RAGTrainer(model_name="FinColBERT", pretrained_model_name="colbert-ir/colbertv2.0")

qa_pairs = list(zip(qa_dataset["input"], qa_dataset["output"]))

print(qa_pairs)

trainer.prepare_training_data(raw_data=qa_pairs, data_out_path="./data/checkpoints/")

trainer.train(batch_size=32)