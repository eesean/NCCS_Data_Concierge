from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from Agent import stream_question_agent

app = FastAPI(title="NCCS Data Concierge API")

class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None
    history: Optional[list] = None

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    return StreamingResponse(
        stream_question_agent(req.question, model=req.model, history=req.history),
        media_type="text/event-stream",
    )