import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

class IntentClassifier:
    def __init__(self, json_path="intent_data.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.exists("intent_index.faiss") and os.path.exists("intent_metadata.json"):
            self.index = faiss.read_index("intent_index.faiss")
            with open("intent_metadata.json", "r") as f:
                self.intent_map = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)
            self.intent_map = []
            self._load_data(json_path)

    def _load_data(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        sentences = []
        for intent_block in data:
            intent = intent_block["intent"]
            for example in intent_block["examples"]:
                sentences.append(example)
                self.intent_map.append(intent)
        self.sentences = sentences 
        embeddings = self.model.encode(sentences)
        self.index.add(np.array(embeddings).astype("float32"))

    def predict_intent(self, message, threshold=0.60):
        vector = self.model.encode([message]).astype("float32")
        D, I = self.index.search(vector, 1)
        score = 1 - D[0][0] ** 0.5
        if score < threshold:
            return "unknown"
        return self.intent_map[I[0][0]]

   
    def get_top_k_examples(self, message, k=3):
        vector = self.model.encode([message]).astype("float32")
        D, I = self.index.search(vector, k)
        results = []
        for idx in I[0]:
            results.append({
                "example": self.sentences[idx],
                "intent": self.intent_map[idx]
            })
        return results



def build_vector_store():
    classifier = IntentClassifier("intent_data.json")
    faiss.write_index(classifier.index, "intent_index.faiss")
    with open("intent_metadata.json", "w") as f:
        json.dump(classifier.intent_map, f)
    print("✅ Vector store and metadata saved successfully.")
