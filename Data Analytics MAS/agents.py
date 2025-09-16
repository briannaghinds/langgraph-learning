"""
@author: Brianna Hinds
Description: Describes/creates all the agents for the data analytics MAS.
"""
from dotenv import load_dotenv
import os
from tools import data_analysis, data_visualization, load_dataset
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.graph import StateGraph, END
import time

load_dotenv()
api_key = os.getenv("G_API_TOKEN")


# define the LLM
# llm = ChatOllama(
#     model="mistral",
#     # model="gpt-oss:20b",
#     temperature=0
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# # bind tools to each agent's LLM
# loader_llm = llm.bind_tools([load_dataset])
# analysis_llm = llm.bind_tools([data_analysis])
# visualization_llm = llm.bind_tools([data_visualization])
supervisor_llm = llm  # normal base llm

# define agent states
class AgentState(TypedDict):
    data_path: str
    dataset: dict
    analysis_results: dict
    visualizations: dict
    final_report: str  # this will be a string in markdown format


## AGENTS ## (agent definitions)
def DataLoaderAgent(state: AgentState) -> AgentState:
    """Agent that loads the dataset from the path, can be a .txt, .csv, .json, or .xlsx file."""

    # call the loading tool for the agent
    response = load_dataset(state["data_path"])
    # response = tool output
    state["dataset"] = response    
    return state

def AnalystAgent(state: AgentState) -> AgentState:
    """Agent that computes the stats, trends, outliers, and any other numerical analysis to the data."""
    response = data_analysis({"data": state["dataset"]})
    state['analysis_results'] = response
    return state

def VisualizationAgent(state: AgentState) -> AgentState:
    """Generate charts and data visualizations for the dataset."""
    response = data_visualization({"data": state["dataset"]})
    state["visualizations"].append(response)
    return state

def SupervisorAgent(state: AgentState) -> AgentState:
    """Supervisor agent orchestrates workflow & writes final report in markdown format."""
    
    analysis_text = str(state["analysis_results"])
    viz_text = str(state["visualizations"])
    response = supervisor_llm.invoke(f"Create a final markdown report.\nAnalysis:\n{analysis_text}\n\nVisualizations:\n{viz_text}")

    # response = llm.invoke(f"Create a final markdown report from the agent's results. \nAnalysis: {state['analysis_results']}\nVisualizations: {state['visualizations']}")
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
    start = time.time()
    # define state
    initial_state = {
        "data_path": "./Data Analytics MAS/data.csv",
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

    # write to a markdown file
    md_file = "final_MAS_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(str(result["final_report"].content))
    f.close()

    print(f"Whole process took {time.time() - start} seconds.")