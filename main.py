import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Sentiment Analysis API (Alternative Version)")

# --- Data Models ---
class SentimentRequest(BaseModel):
    text: str = Field(..., example="I absolutely love this new product, it's fantastic!")

class SentimentResponse(BaseModel):
    sentiment: str
    explanation: str


# Helper function to generate sentiment analysis
def analyze_with_llm(text: str):
    prompt = f"""
You are a sentiment analysis model.
Classify the text as "positive", "negative", or "neutral".
Return ONLY valid JSON in this format:

{{
  "sentiment": "positive | negative | neutral",
  "explanation": "your explanation here"
}}

Text: "{text}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",    # you can change model
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Extract the JSON content
    try:
        result_json = response.choices[0].message.content
        return eval(result_json)  # safe here because model outputs pure JSON
    except Exception as e:
        raise ValueError("Invalid JSON returned by the model.")


# --- API Endpoint ---
@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = analyze_with_llm(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Welcome to the Alternate Sentiment API. POST to /analyze-sentiment."}


# For running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
