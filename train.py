from ragatouille import RAGTrainer
from datasets import load_dataset

qa_dataset = load_dataset("FinGPT/fingpt-fiqa_qa")

trainer = RAGTrainer(model_name = "FinColBERT", pretrained_model_name = "colbert-ir/colbertv2.0")

trainer.prepare_training_data(raw_data=qa_dataset, data_out_path="./data/checkpoints/")

trainer.train(batch_size=32)

