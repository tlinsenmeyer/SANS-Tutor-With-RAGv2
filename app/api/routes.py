from fastapi import APIRouter
from pydantic import BaseModel
from app.graph.tutor_graph import build_tutor_graph
from app.graph.state import TutorState

router = APIRouter()

# Request model
class TutorRequest(BaseModel):
    question: str
    user_answer: str | None = None

# Build graph once at startup
graph = build_tutor_graph()

@router.post("/tutor")
async def run_tutor_pipeline(req: TutorRequest):
    """
    Run the full tutor pipeline:
    - retrieval
    - tutor explanation
    - grading
    """
    # Initialize state
    state = TutorState(
        question=req.question,
        user_answer=req.user_answer
    )

    # Run graph
    final_state = graph.invoke(state)

    # Return final state as JSON
    return final_state.dict()
