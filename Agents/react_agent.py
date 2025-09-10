# library imports
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages  # reducer function
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

"""NOTES
Annotated: gives extra context without changing the data type
    email = Annotated[str, "This has to be a valid email format"]
Sequence: reducer function, updates the nodes by combining existing states (DOESNT OVERRIDE EXISTING DATA)

GRAPH STRUCTURE: Start -> Agent -continue-> Tools (loop) -> Agent -end-> END 
"""


"""TYPES OF MESSAGES
BaseMessage: foundational class for all message types in LangGraph
ToolMessage: passes data back from the LLM after it calls a tool
SystemMessage: gives instructions to the LLM
"""

# ## load environment variables
# load_dotenv()
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ## hugging face client
# client = InferenceClient(
#     "togethercomputer/GPT-NeoXT-Chat-Base-20B", # gpt-oss20b model
#     token=HF_API_TOKEN
# )


# define the AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # this saves the state by appending onto the message


# define the tools for the agent
@tool 
def add(a:int, b:int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

tools = [add]

# define the model
# temperture = 0 minimizes hallucination
model = ChatOllama(model="gpt-oss:20b", temperature=0).bind_tools(tools)  # bind the tool list to the model so it knows what tools is can use 

# create a node that will act as the agent within the graph workflow
def model_call(state: AgentState) -> AgentState:
    # define the prompt for the model
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])  # feed the prompt through the model
    return {"messages": [response]}

# this method will help determine if the agent needs to loop or end the workflow
def should_continue(state: AgentState):  # not need to return the AgentState
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

# define the graph and its nodes
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

# build the graph
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

application = graph.compile()


## NOT RELATED TO LANGGRAPH ##
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]  # get recent message
        
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
####


## MAIN ##
if __name__ == "__main__":
    inputs = {"messages": [("user", "Add 40 + 12.")]}
    print_stream(application.stream(inputs, stream_mode="values"))
