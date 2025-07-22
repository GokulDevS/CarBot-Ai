from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re  
from llm_utils import predict_intent_llm

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageInput(BaseModel):
    message: str

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)         
    return text.lower().strip()

@app.post("/predict-intent")
async def predict_intent(input: MessageInput):
    user_message = input.message
    cleaned_message = clean_text(user_message) 
    result = predict_intent_llm(cleaned_message) 
    return result
