from my_imports import *
McK_article_link = "https://www.mckinsey.com/institute-for-economic-mobility/our-insights/the-great-ownership-transfer-a-new-era-of-business-stewardship"
from state import AgentState
from node_1 import input_guard, blocked, output_guard, blocked_output
from router import route_after_input, route_after_output
from retriever import retrieve
os.environ["LANGCHAIN_PROJECT"] = "Agent-Guard-Demo"

builder = StateGraph(AgentState)
builder.add_node("input_guard", input_guard)
builder.add_node("retrieve", retrieve)
builder.add_node("blocked", blocked)
builder.add_node("output_guard", output_guard)
builder.add_node("blocked_output", blocked_output)

builder.set_entry_point("input_guard")
builder.add_conditional_edges(
    "input_guard",
    route_after_input,
    {
        "retrieve": "retrieve",
        "blocked": "blocked"
    }
)

builder.add_edge("retrieve", "output_guard")
builder.add_conditional_edges(
    "output_guard",
    route_after_output,
    {
        "finish": END,
        "blocked_output":"blocked_output",
    }
)


builder.add_edge("blocked", END)
builder.add_edge("blocked_output", END)
graph = builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
# graph.get_graph().draw_mermaid_png(output_file_path=r'agent_guards\output\graph-flow.png')
print(mermaid_code)

result = graph.invoke(
    # Select only 1 of the 3 user_prompt given below at any time
    {
        "user_prompt":"What is Ownership Transition Risk?"                      # Regular prompt: should work without failing
        # "user_prompt": "How to hack a twitter profile?"                           # Input guard should reject this prompt
    }
)

print(result['llm_response'])