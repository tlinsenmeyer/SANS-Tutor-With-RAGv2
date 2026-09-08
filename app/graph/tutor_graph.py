from langgraph.graph import StateGraph, END
from app.graph.state import TutorState

# Placeholder agent functions — you will implement these later
def retrieve_agent(state: TutorState) -> TutorState:
    # TODO: call Qdrant, retrieve documents
    state.retrieved_docs = ["doc1", "doc2"]  # placeholder
    return state

def tutor_agent(state: TutorState) -> TutorState:
    # TODO: call vLLM, generate explanation
    state.tutor_response = "This is a placeholder tutor response."
    return state

def grading_agent(state: TutorState) -> TutorState:
    # TODO: call vLLM, evaluate answer
    state.grade = "Correct"
    state.score = 0.95
    return state

# Build the graph
def build_tutor_graph():
    graph = StateGraph(TutorState)

    # Add nodes (agents)
    graph.add_node("retrieve", retrieve_agent)
    graph.add_node("tutor", tutor_agent)
    graph.add_node("grade", grading_agent)

    # Define workflow edges
    graph.add_edge("retrieve", "tutor")
    graph.add_edge("tutor", "grade")
    graph.add_edge("grade", END)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Compile graph
    return graph.compile()
