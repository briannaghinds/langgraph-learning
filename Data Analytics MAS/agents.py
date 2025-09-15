"""
@author: Brianna Hinds
Description: Describes/creates all the agents for the data analytics MAS.
"""
from tools import *
from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.graph import StateGraph, END

# define the LLM
llm = ChatOllama(
    model="mistral",
    temperature=0
)

# bind tools to each agent's LLM
loader_llm = llm.bind_tools([load_dataset])
analysis_llm = llm.bind_tools([data_analysis])
visualization_llm = llm.bind_tools([data_visualization])
supervisor_llm = llm  # normal base llm

# define agent states
class AgentState(TypedDict):
    dataset_path: str
    dataset: dict
    analysis_results: dict
    visualizations: list[str]
    final_report: str  # this will be a string in markdown format


## AGENTS ## (agent definitions)
def DataLoaderAgent(state: AgentState) -> AgentState:
    """Agent that loads the dataset."""

    # ask the loader_llm to call its tool
    response = loader_llm.invoke(f"Load dataset from {state['dataset_path']}")

    # response = tool output
    state["dataset"] = response    
    return state

def AnalystAgent(state: AgentState) -> AgentState:
    """Agent that computes the stats, trends, outliers, in the data."""
    response = analysis_llm.invoke("Analyze the dataset and return key statistics, facts, numbers, and anything that seems relevant.", input={"data": state['dataset']})
    state['analysis_results'] = response
    return state

def VisualizationAgent(state: AgentState) -> AgentState:
    """Agent responsible for creating charts and data visualizations."""
    response = visualization_llm.invoke("Generate visualizations of dataset trends, export the .png images into a folder name /graphs.", input={"data": state['dataset']})
    state["visualizations"].append(response)
    return state

def SupervisorAgent(state: AgentState) -> AgentState:
    """Supervisor agent orchestrates workflow & writes final report in markdown format."""
    
    response = llm.invoke(f"Create a final markdown report from the agent's results. \nAnalysis: {state['analysis_results']}\nVisualizations: {state['visualizations']}")
    state["final_report"] = response
    return state


## GRAPH ##
# define the graph and its nodes
graph = StateGraph(AgentState)
graph.add_node("loader", DataLoaderAgent)
graph.add_node("analyst", AnalystAgent)
graph.add_node("visualize", VisualizationAgent)
graph.add_node("supervisor", SupervisorAgent)

# define edges (workflow order)  NOTE: this workflow seems very linear
graph.add_edge("loader", "analyst")
graph.add_edge("analyst", "visualize")
graph.add_edge("visualize", "supervisor")
graph.add_edge("supervisor", END)
graph.set_entry_point("loader")
workflow = graph.compile()


## MAIN ##
if __name__ == "__main__":
    # define state
    initial_state = {
        "dataset_path": "",
        "dataset": None,
        "analysis_results": None,
        "visualizations": [],
        "final_report": ""
    }


    # invoke agent
    result = workflow.invoke(
        initial_state
    )

    print(result["final_report"])