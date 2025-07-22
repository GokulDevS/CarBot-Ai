import os
import json
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("intent_index.faiss")

with open("intent_metadata.json", "r") as f:
    intent_map = json.load(f)

with open("intent_data.json", "r") as f:
    intent_data = json.load(f)

with open("intent_responses.json", "r") as f:
    intent_responses = json.load(f)


def retrieve_top_k_examples(message, k=5):
    query_vec = model.encode([message]).astype("float32")
    D, I = index.search(query_vec, k)

    examples = []
    for idx in I[0]:
        if idx < len(intent_map):
            examples.append({
                "intent": intent_map[idx],
                "example": intent_data[idx // 20]["examples"][idx % 20]  # assumes 20 examples per intent
            })
    return examples


def build_prompt(message, top_k_examples):
    examples_text = "\n".join([f"- \"{e['example']}\" → {e['intent']}" for e in top_k_examples])
    return f"""You are an intent classifier assistant.
Based on the following user message and previous labeled examples, identify the most appropriate intent.

User message: "{message}"

Here are some labeled training examples:
{examples_text}

Respond with the best matching intent only (e.g., 'book', 'cancel', or 'greet').
If no intent matches well, respond with 'unknown'."""


import re

def query_groq_llm(prompt):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful intent classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
    )

    result = response.json()
    raw_output = result["choices"][0]["message"]["content"].strip().lower()

 
    match = re.search(r"\b(book|cancel|greet|unknown)\b", raw_output)
    return match.group(1) if match else "unknown"

def predict_intent_llm(message):
    top_k = retrieve_top_k_examples(message)
    prompt = build_prompt(message, top_k)
    predicted_intent = query_groq_llm(prompt)
    
    response = intent_responses.get(predicted_intent, "Sorry, I couldn't understand your request.")
    return {
        "intent": predicted_intent,
        "response": response
    }
