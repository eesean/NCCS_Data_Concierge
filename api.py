from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict

from pipeline import handle_question  

app = FastAPI(title="NCCS Data Concierge API")

class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    return handle_question(req.question, model=req.model)