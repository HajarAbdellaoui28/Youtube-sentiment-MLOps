
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

MODEL_PATH = Path("models/sentiment_pipeline.joblib")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Modèle introuvable")
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(
    title="YouTube Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

class BatchRequest(BaseModel):
    comments: list[str] = Field(min_length=1, max_length=500)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible"
        )

    comments = request.comments

    if any(not text.strip() for text in comments):
        raise HTTPException(
            status_code=422,
            detail="Les commentaires vides sont interdits"
        )

    predictions = model.predict(comments).tolist()

    labels = {
        -1: "negative",
        0: "neutral",
        1: "positive"
    }

    results = [
        {
            "text": text,
            "label": int(pred),
            "sentiment": labels[int(pred)]
        }
        for text, pred in zip(comments, predictions)
    ]

    stats = {
        sentiment: sum(
            r["sentiment"] == sentiment for r in results
        )
        for sentiment in labels.values()
    }

    return {
        "total": len(results),
        "results": results,
        "statistics": stats
    }
