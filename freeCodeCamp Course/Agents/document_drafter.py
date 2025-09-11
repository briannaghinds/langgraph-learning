"""GOAL
Create an AI agent that given a prompt (system prompt + context) will draft an email/letter.
"""

# import libraries
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages  # reducer function (adds to the state instead of replacing)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# GLOBAL VARIABLE
document_content = ""  # this is just a simple way to hold the document content


# define drafter state
class DrafterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 


# define tools (DONT FORGET THE DOCSTRINGS)
@tool
def update(context: str) -> str:
    """
    Updates the document with the provided context.

    Args:
        context: content of the document
    """ 

    global document_content
    document_content = context

    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """
    Save the current document to a text file and finish the drafting process

    Args:
        filename: name for the text file
    """

    global document_content

    # ROBUSTNESS CONDITION (just to make sure our filename ends in .txt)
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    # open, write, and save drafted document
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"~ Document has been saved to {filename}")
        return f"~ Document has been saved to {filename}"
    except Exception as e:
        return f"~Error saving document: {str(e)}"
    

tools = [update, save]
model = ChatOllama(model="gpt-oss:20b", temperature=0)  # temperture set to 0 so model can be more deterministic (no hallucinations)

# define agent
def drafter_agent(state: DrafterState) -> DrafterState:
    # system prompt should be written in as a SystemMessage (gives instructions to LLM)
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    # ROBUSTNESS CHECK
    if not state["messages"]:  # this is the initial chat to start off the conversation (checking if NO messages exists)
        initial_chat = "I'm ready to help you update a document. WHat would you like to create? "
        user_message = HumanMessage(content=initial_chat)
    else:
        chat = "What would you like to do with the document? "
        print(f"USER: {chat}")
        user_message = HumanMessage(content=chat)

    # give the full context for the LLM in list format
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    model_response = model.invoke(all_messages)
    print(f"AI: {model_response.content}")

    ## PRETTY PRINT TOOL CALLS
    if hasattr(model_response, "tool_calls") and model_response.tool_calls:
        print(f"~USING TOOLS: {[tc["name"] for tc in model_response.tool_calls]}")

    # return the new state
    return {"messages": list(state["messages"]) + [user_message + model_response]}


# define a function to determine the next node to go to (used in conditional edge call)
def should_continue(state: DrafterState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    # ROBUST CHECK: continue the next node if there is nothing to save
    if not messages:
        return "continue"
    
    # check most recent tool message
    for message in reversed(messages):
        # check if message is a ToolMessage (it came from the tool node)
        # CHECK: is a ToolMessage type, has "saved" and "document" in the message in same way, whatever order
        if (isinstance(message, ToolMessage)) and ("saved" in message.content.lower()) and ("document" in message.content.lower()):
            return "end"  # end node
        
    return "continue"


## PRETTY PRINT METHOD ##
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"~TOOL RESULT: {message.content}")
####

# initialize and build the graph and its nodes
graph = StateGraph(DrafterState)
graph.add_node("agent", drafter_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)
draft_app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in draft_app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")


## MAIN ##
if __name__ == "__main__":
    run_document_agent()