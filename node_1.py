from my_imports import *
import json
from state import AgentState
from pydantic import BaseModel


class inputResult(BaseModel):
    safe:bool = Field(description="Whether the response is safe")
    reason:str = Field(description="Reasoning behind the classification in short (max 40 words)")

class outputResult(BaseModel):
    safe:bool = Field(description="Whether the response is safe")
    relevant:bool = Field(description="Whether the response is relevant")

# guard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
guard_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

input_guard_llm = guard_llm.with_structured_output(inputResult)
output_guard_llm = guard_llm.with_structured_output(outputResult)

def input_guard(state:AgentState):
    prompt = f"""
                You are a safety classifier. Analyze the user prompt and determine if it is safe

                User Prompt:{state['user_prompt']}
            """
    
    result = input_guard_llm.invoke(
        [HumanMessage(content=prompt)]
    )
    state["input_safe"] = result.safe
    state['reason'] = result.reason
    return state

def blocked(state):
    reason = state.get('reason', "Unknown Reason")
    state['llm_response'] = f"Prompt Rejected. Reason: {reason}"
    return state

def output_guard(state):
    prompt = f"""
            Check if Response is safe and relevant to the User Prompt
            Return only JSON
                {{
                "safe": true,
                "relevant":true,
                }}
                Response: {state['llm_response']}
                User Prompt:{state['user_prompt']} 
            """
    result = output_guard_llm.invoke(
        [HumanMessage(content=prompt)]
    )
    # result = json.loads(output_guard_response.content)
    state['output_safe'] = result.safe and result.relevant
    return state

def blocked_output(state):
    state['llm_response'] = (
        "Output Rejected. Generated Response either voilated policy or was not relevant"
    )
    return state