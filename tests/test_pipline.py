import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.graph.state import TutorState
from app.agents.tutor_agent import tutor_agent
from app.agents.verification_agent import grading_agent

state = TutorState(
    question="What is the purpose of a firewall?",
    user_answer="It blocks bad traffic."
)

state = tutor_agent(state)
print("\n--- Tutor Response ---")
print(state.tutor_response)

state = grading_agent(state)
print("\n--- Grading Output ---")
print("Grade:", state.grade)
print("Score:", state.score)
