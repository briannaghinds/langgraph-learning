"""
This is a simple LLM bot, where it has no memory of past statements/conversations.
This is the simpliest chat bot that can be made via LangGraph.

GRAPH STRUCTURE
Start -> Process Node -> End
"""

# LLM stuff
from langchain_ollama import ChatOllama

# langgraph imports
from typing import TypedDict
from langchain_core.messages import HumanMessage  # messages passed from human to model
from langgraph.graph import StateGraph, START, END


## UPLOAD GPT-OSS MODEL  ##
llm = ChatOllama(model="gpt-oss:20b")
####

# define the AgentState
class AgentState(TypedDict):
    messages: list[HumanMessage]

# define the tool node
def process(state: AgentState) -> AgentState:
    """This node takes the user's message and creates a response"""
    last_message = state["messages"][-1]
    response = llm.invoke(last_message.content)  # pass the user input to the LLM and give a response

    print(f"AI: {response.content}")
    return state


# initialize the graph and its nodes
workflow = StateGraph(AgentState)
workflow.add_node("AI", process)

# build the graph and compile
workflow.add_edge(START, "AI")
workflow.add_edge("AI", END)
chat_bot = workflow.compile()


# get user input
if __name__ == "__main__":
    while True:
        user_input = input("Enter: ").lower()
        if user_input in ["exit", "quit"]:
            print("Exiting chatbot.")
            break

        # invoke chat_bot via LangGraph
        chat_bot.invoke({"messages": [HumanMessage(content=user_input)]})