from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict

from pipeline import handle_question_agent, stream_question_agent

app = FastAPI(title="NCCS Data Concierge API")

class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    return handle_question_agent(req.question, model=req.model)

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    return StreamingResponse(
        stream_question_agent(req.question, model=req.model),
        media_type="text/event-stream",
    )