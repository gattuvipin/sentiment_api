# Sentiment Analysis API

A simple REST API built with FastAPI, OpenAI, and Docker that analyzes the sentiment of text inputs using an LLM.

## Project Structure

- `main.py` — The FastAPI application and LangChain logic.  
- `Dockerfile` — Configuration for containerizing the application.  
- `requirements.txt` — Python dependencies.

## Prerequisites

- Docker installed on your machine.  
- An OpenAI API Key (or another LLM provider supported by LangChain).

## 1. Local Setup (Without Docker)

If you want to test the code while developing:

Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure environment:

Create a `.env` file in the project root and add your key:
```env
OPENAI_API_KEY=sk-your-key-here
```

Run the app:
```bash
uvicorn main:app --reload
```

Access the API docs at: http://localhost:8000/docs

## 2. Docker Setup (Recommended)

Build the image:
```bash
docker build -t sentiment-api .
```

Run the container (pass your API key into the container):
```bash
docker run -p 8000:8000 --env-file .env sentiment-app
```

Test the API:

Open http://localhost:8000/docs in your browser to use the interactive Swagger UI, or use curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze-sentiment' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I really enjoyed using this tool, it made my life so much easier!"
}'
```

Expected output:
```json
{
  "sentiment": "positive",
  "explanation": "The user expresses enjoyment and states that the tool made their life easier."
}
```

## 3. Git Workflow

Initialize Git:
```bash
git init
```

Add files:
```bash
git add .
```

Commit changes:
```bash
git commit -m "Initial commit: Setup FastAPI, LangChain and Docker"
```

Push to GitHub (after creating a repo on GitHub):
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentiment-api.git
git push -u origin main
```