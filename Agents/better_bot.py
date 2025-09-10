"""
This chatbot will have a memory, however as the memory grows (past a lenght of 5 for now) the memory will need to be cleaned out.
GRAPH STRUCTURE:
START -> Process -> END
"""

# this is a chatbot that has memory
import os
from langchain_ollama import ChatOllama
from typing import TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# define llm
llm = ChatOllama(model="gpt-oss:20b")

# define agent's state
class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]


# define the tool 
def process(state: AgentState) -> AgentState:
    """This node will solve the request from the user's input"""
    response = llm.invoke(state["messages"])  # get the llms response to the user request
    state["messages"].append(AIMessage(content=response.content))  # append the new state
    print(f"AI: {response.content}")

    return state

# initalize the graph and node
workflow = StateGraph(AgentState)
workflow.add_node("conversation", process)

# build the graph
workflow.add_edge(START, "conversation")
workflow.add_edge("conversation", END)
chatbot = workflow.compile()


## MAIN ##
if __name__ == "__main__":
    conversation_history = []
    print("Enter 'exit' or 'end' to exit the conversation.")

    while True:
        user_input = input("You: ")

        # conversation_history.append(HumanMessage(content=user_input))
        if user_input.lower() in ["exit", "end"]:
            break

        conversation_history.append(HumanMessage(content=user_input))
        result = chatbot.invoke({"messages": conversation_history})
        conversation_history = result["messages"]

        # add user's input into the text file
        with open("./Agents/better_bot_memory.txt", "w") as memory:
            memory.write("YOUR CONVERSATION LOG:\n")
            
            for message in conversation_history:
                if isinstance(message, HumanMessage):
                    memory.write(f"You: {message.content}")
                elif isinstance(message, AIMessage):
                    memory.write(f"AI: {message.content}")
            memory.write("END OF CONVERSATION")
        print("~Conversation has been saved to better_bot_memory.txt")

    memory.close()
