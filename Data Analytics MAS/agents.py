"""
@author: Brianna Hinds
Description: Describes/creates all the agents for the data analytics MAS.
"""

from langchain_ollama import ChatOllama
from typing import TypedDict

# define the LLM
llm = ChatOllama(
    model="mistral",
    temperature=0
)

# define agent states
class AgentState(TypedDict):
    dataset: dict
    analysis_results: dict
    visualizations: list[str]
    final_report: str  # this will be a string in markdown format


## AGENTS ## (agent definitions)
def DataLoaderAgent(state: AgentState) -> AgentState:
    """Agent that loads the dataset."""
    # TOOL: load_dataset
    pass

def AnalystAgent(state: AgentState) -> AgentState:
    """Agent that computes the stats, trends, outliers, in the data."""
    pass

def VisualizationAgent(state: AgentState) -> AgentState:
    """Agent responsible for creating charts and data visualizations."""
    pass

def SupervisorAgent(state: AgentState) -> AgentState:
    """Supervisor agent orchestrates workflow & writes final report in markdown format."""
    pass