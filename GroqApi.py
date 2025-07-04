from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat/")
async def chat_endpoint(message: Message):
    conversation = [
        {"role": "system", "content": "Eres un asistente útil y amigable."},
        {"role": "user", "content": message.text}
    ]

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=conversation,
        max_tokens=150
    )

    reply = response.choices[0].message.content
    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
