"""
@author: Brianna Hinds
Description: Agent definitions, graph initialization, and main run of the Financial Fraud MAS.
"""

# import libraries
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from tools import *
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

# define llm
load_dotenv()
GOOGLE_API_KEY = os.getenv("G_API_TOKEN")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

system_prompt = """
You are the ingestion agent. Your task is to gather transaction data from all sources in 'transaction_data_sources'.
Do not stop until all sources are loaded into 'transaction_data'. Return only the updated data in the specified format.
"""  # might put this in the .txt file in .gitignore


# bind the specific tools to each llm
ingestion_llm = llm.bind_tools([load_csv, fetch_api, query_database], system_prompt=system_prompt)
transaction_analysis_llm = llm.bind_tools([statistical_summary, time_series_analysis, outlier_detection])
fraud_detection_llm = llm.bind_tools([ml_fraud_model, rule_based_checker, graph_based_detection])
risk_classification_agent = llm.bind_tools([risk_scorer, regulatory_checker])
supervisor_llm = llm  # the supervisor has no tools


# define agent state
class AgentState(TypedDict):
    transaction_data_sources: list[str]  # there will be multiple sources (paths, utls, db connections, etc)
    transaction_data: dict  # unified list of transaction records
    fraud_data: list[dict]  # flagged anomalies/vulnerabilities
    risk_report: str  # final summary/report (output in pdf format and in json formats)


# define agents
def ingestion_agent(state: AgentState) -> AgentState:
    """Agent that loads the data from the AgentState until it has the complete dataset."""
    response = ingestion_llm.invoke({"datapath": state["transaction_data_sources"]})
    state["transaction_data"] = response

    return state

def transaction_analysis_agent(state: AgentState) -> AgentState:
    """Agent that goes through the transaction data, parse for trends, anomalies, patterns, and get a readable analysis report."""
    return state

def fraud_detection_agent(state: AgentState) -> AgentState:
    """Detect potential fradulent transactions from the inputted transaction data."""
    return state

def risk_analysis_agent(state: AgentState) -> AgentState:
    """The risk analysis agent will go through each transaction and define a risk score."""
    return state

def supervisor_agent(state: AgentState) -> AgentState:
    """Supervisor agent orchestrates the fraud and risk report and exports it into a PDF compilance report and JSON dashboard data."""
    return state


# initalize graph and its nodes
workflow = StateGraph(AgentState)
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("transaction_analysis", transaction_analysis_agent)
workflow.add_node("fraud_detector", fraud_detection_agent)
workflow.add_node("risk_analysis", risk_analysis_agent)
workflow.add_node("supervisor", supervisor_agent)

# build graph
workflow.set_entry_point("ingestion")

fraud_detector = workflow.compile()

# visualize the graph
# extra step, but this visualizes the graph we created in LangGraph
display(Image(fraud_detector.get_graph().draw_mermaid_png()))

## MAIN ##
if __name__ == "__main__":
    initial_state = {
        "transaction_data_sources": [],
        "transaction_data": {},
        "fraud_data": [],
        "risk_report": ""

    }

    result = fraud_detector.invoke(initial_state)
    print(result["risk_report"])