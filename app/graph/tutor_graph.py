# app/graph/tutor_graph.py

from langgraph.graph import StateGraph, END
from app.graph.state import TutorState
from app.agents.retrieval_agent import retrieval_agent
from app.agents.tutor_agent import tutor_agent
from app.agents.verification_agent import verification_agent

def build_tutor_graph():
    graph = StateGraph(TutorState)

    graph.add_node("retrieve", retrieval_agent)
    graph.add_node("tutor", tutor_agent)
    graph.add_node("verify", verification_agent)

    graph.add_edge("retrieve", "tutor")
    graph.add_edge("tutor", "verify")
    graph.add_edge("verify", END)

    graph.set_entry_point("retrieve")

    return graph.compile()
