from transformers import pipeline

prompt = f"""Please write a scientific paper passage to answer the question
Question: what was the s&p 500 looking like in 2008
Passage:"""

messages = [
    {"role": "user", "content": prompt},
]
pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
pipe(messages)