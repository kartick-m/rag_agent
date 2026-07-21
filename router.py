from state import AgentState

def route_after_input(state):
    if state['input_safe']:
        return "retrieve"
    return "blocked"


def route_after_output(state):
    if state['output_safe']:
        return "finish"
    return "blocked_output"