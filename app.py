from my_imports import *
from flask import Flask, request, render_template

McK_article_link = "https://www.mckinsey.com/institute-for-economic-mobility/our-insights/the-great-ownership-transfer-a-new-era-of-business-stewardship"
from state import AgentState
from node_1 import input_guard, blocked, output_guard, blocked_output
from router import route_after_input, route_after_output
from retriever import retrieve

os.environ["LANGCHAIN_PROJECT"] = "Agent-Guard-Demo"


def build_graph():
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
            "blocked": "blocked",
        },
    )

    builder.add_edge("retrieve", "output_guard")
    builder.add_conditional_edges(
        "output_guard",
        route_after_output,
        {
            "finish": END,
            "blocked_output": "blocked_output",
        },
    )

    builder.add_edge("blocked", END)
    builder.add_edge("blocked_output", END)
    return builder.compile()


graph = build_graph()

def answer_question(question: str) -> str:
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return "Please enter a question."

    try:
        result = graph.invoke({"user_prompt": cleaned_question})
        return result.get("llm_response", "No answer was generated.")
    except Exception as exc:
        return f"Sorry, I couldn't process that request right now: {exc}"


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", answer=None, question="")

    @app.post("/")
    def submit_question():
        question = request.form.get("question", "")
        answer = answer_question(question)
        return render_template("index.html", answer=answer, question=question)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)