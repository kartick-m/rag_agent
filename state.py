from typing import TypedDict

class AgentState(TypedDict):
    user_prompt: str
    llm_response: str
    input_safe: bool
    output_safe: bool
    reason: str
    retrieved_contexts: list[str]

