# app/agents/tutor_agent.py

from app.llm.vllm_client import VLLMClient
from app.graph.state import TutorState

llm = VLLMClient()

async def tutor_agent(state: TutorState) -> TutorState:
    """
    Tutor agent:
    - Uses merged context (Qdrant + Neo4j placeholder)
    - Produces a structured, SANS-style explanation
    """

    if not state.question:
        raise ValueError("State.question is empty — cannot generate tutor response.")

    system_prompt = (
        "You are a cybersecurity tutor trained in SANS-style instruction.\n"
        "Your job:\n"
        "- Use the provided context when relevant.\n"
        "- Do NOT invent citations.\n"
        "- If context is missing or incomplete, say so.\n"
        "- Provide a clear, structured explanation.\n"
        "- Keep reasoning internal; output only the final explanation.\n"
    )

    user_content = (
        f"Question:\n{state.question}\n\n"
        f"Context:\n{state.merged_context or '[no context available]'}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = await llm.chat(messages)

    state.tutor_response = response
    return state
