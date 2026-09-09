# app/agents/verification_agent.py

from app.graph.state import TutorState
from app.llm.vllm_client import VLLMClient
import json

llm = VLLMClient()

async def verification_agent(state: TutorState) -> TutorState:
    """
    Verification agent:
    - checks the tutor_response for accuracy, relevance, and completeness
    - compares it against retrieved context (Qdrant + Neo4j placeholder)
    - adds modern updates beyond the SANS manuals
    - produces a final improved answer + confidence score
    """

    if not state.tutor_response:
        raise ValueError("State.tutor_response is empty — cannot verify.")

    system_prompt = (
        "You are a cybersecurity verification and enrichment agent.\n"
        "You receive:\n"
        "- the user's question\n"
        "- the tutor's answer\n"
        "- retrieved SANS/graph context\n\n"
        "Your job:\n"
        "1) Check the tutor answer for factual accuracy, relevance, and completeness.\n"
        "2) Compare it against the provided context.\n"
        "3) Identify any outdated or incorrect information.\n"
        "4) Add modern updates (new threats, mitigations, standards) beyond the SANS manuals.\n"
        "5) Produce a final improved answer.\n"
        "Respond ONLY in JSON with keys:\n"
        "verified_answer, corrections, additions, modern_updates, final_answer, confidence."
    )

    user_content = (
        f"Question:\n{state.question}\n\n"
        f"Tutor Answer:\n{state.tutor_response}\n\n"
        f"Context (Qdrant + Neo4j):\n{state.merged_context or '[no context available]'}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw_output = await llm.chat(messages)

    try:
        result = json.loads(raw_output)
    except Exception:
        # fallback: treat whole output as final_answer
        result = {
            "verified_answer": state.tutor_response,
            "corrections": [],
            "additions": [],
            "modern_updates": [],
            "final_answer": raw_output,
            "confidence": 0.5,
        }

    state.verified_answer = result.get("verified_answer")
    state.corrections = result.get("corrections", [])
    state.additions = result.get("additions", [])
    state.modern_updates = result.get("modern_updates", [])
    state.final_answer = result.get("final_answer", state.tutor_response)
    state.confidence = result.get("confidence", 0.5)

    return state
