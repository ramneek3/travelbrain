"""TravelBrain web server. Run: .venv/bin/uvicorn app:app --reload"""
import json
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agents
import db

app = FastAPI(title="TravelBrain")
STATIC = Path(__file__).parent / "static"
db.init()


class PlanRequest(BaseModel):
    prompt: str


class ChatRequest(BaseModel):
    message: str


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _error_text(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY and restart the server."
    if isinstance(e, openai.RateLimitError):
        return "Free-model rate limit reached on OpenRouter. Wait a minute (or until tomorrow for the daily cap) and try again."
    if isinstance(e, openai.APIStatusError):
        return f"OpenRouter error ({e.status_code}): {e.message}"
    if isinstance(e, openai.APIConnectionError):
        return "Could not reach OpenRouter. Check your connection."
    return str(e)


@app.post("/api/plan")
async def plan(req: PlanRequest):
    async def stream():
        try:
            async for event in agents.plan_trip(req.prompt):
                if event["type"] == "plan":
                    reports = {k: {"report": v["report"], "sources": v.get("sources", [])}
                               for k, v in event["reports"].items()}
                    event["id"] = db.save_trip(req.prompt, event["trip"], reports, event["plan"])
                    event["reports"] = reports
                yield sse(event)
        except Exception as e:
            yield sse({"type": "error", "message": _error_text(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/trips")
def trips():
    return db.list_trips()


@app.get("/api/trips/{trip_id}")
def trip(trip_id: str):
    t = db.get_trip(trip_id)
    if not t:
        raise HTTPException(404, "Trip not found")
    return t


@app.delete("/api/trips/{trip_id}")
def delete(trip_id: str):
    db.delete_trip(trip_id)
    return {"ok": True}


@app.post("/api/trips/{trip_id}/chat")
async def chat(trip_id: str, req: ChatRequest):
    saved = db.get_trip(trip_id)
    if not saved:
        raise HTTPException(404, "Trip not found")

    async def stream():
        answer = ""
        try:
            async for chunk in agents.follow_up(saved, req.message):
                answer += chunk
                yield sse({"type": "delta", "text": chunk})
        except Exception as e:
            yield sse({"type": "error", "message": _error_text(e)})
            return
        db.add_message(trip_id, "user", req.message)
        db.add_message(trip_id, "assistant", answer)
        yield sse({"type": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
