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
import time

# define llm
load_dotenv()
GOOGLE_API_KEY = os.getenv("G_API_TOKEN")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

## PROMPTS ##
INGESTION_PROMPT = """
You are the ingestion agent. Your task is to gather transaction data from all sources in 'transaction_data_sources'.
Do not stop until all sources are loaded into 'transaction_data'. Return only the updated data in the specified format.
""" 

TRANSACTION_ANALYSIS_PROMPT = """
Analyze transaction data for trends, anomalies, and patterns.
Return a structured summary of statistical and time series analysis, including outliers.
"""

FRAUD_DETECTION_PROMPT = """
Detect potentially fraudulent transactions using ML models, rule-based checks, and graph-based analysis.
Add flags and scores to each transaction where necessary.
"""

RISK_ANALYSIS_PROMPT = """
Assign risk scores to suspicious transactions and perform regulatory checks.
Return a combined 'risk_report' for all transactions.
"""

SUPERVISOR_PROMPT = """
Combine the results from all agents into a final risk report.
Export as JSON and prepare for PDF output if needed.
"""

####

# bind the specific tools to each llm
ingestion_llm = llm.bind_tools([load_csv, fetch_api, query_database], system_prompt=INGESTION_PROMPT)
transaction_analysis_llm = llm.bind_tools([statistical_summary, time_series_analysis, outlier_detection], system_prompt=TRANSACTION_ANALYSIS_PROMPT)
fraud_detection_llm = llm.bind_tools([ml_fraud_model, rule_based_checker, graph_based_detection], system_prompt=FRAUD_DETECTION_PROMPT)
risk_classification_llm = llm.bind_tools([risk_scorer, regulatory_checker], system_prompt=RISK_ANALYSIS_PROMPT)
supervisor_llm = llm  # the supervisor has no tools


# define agent state
class AgentState(TypedDict):
    transaction_data_sources: list[str]  # there will be multiple sources (paths, utls, db connections, etc)
    transaction_data: dict  # unified list of transaction records
    fraud_data: list[dict]  # flagged anomalies/vulnerabilities
    risk_report: str  # final summary/report (output in pdf format and in json formats)
    final_report: str

# define agents
def ingestion_agent(state: AgentState) -> AgentState:  # this definition allows the agent to rununtil all the data is loaded, regardless of the source type
    """Agent that loops until all transaction sources are ingested."""
    all_data = []
    for source in state["transaction_data_sources"]:
        response = ingestion_llm.invoke({"datapath": source})
        if "error" in response:
            print(f"Error loading {source}: {response['error']}")
        else:
            all_data.extend(response.get("data", []))

    state["transaction_data"] = {"data": all_data}
    return state

def transaction_analysis_agent(state: AgentState) -> AgentState:
    """Agent that goes through the transaction data, parse for trends, anomalies, patterns, and get a readable analysis report. Performs stars, time series, and outlier analysis."""
    data = state["transaction_data"]
    analysis_results = {}
    analysis_results.update(transaction_analysis_llm.invoke({"data": data.get("data", [])}))
    state["transaction_data"]["analysis"] = analysis_results

    return state

def fraud_detection_agent(state: AgentState) -> AgentState:
    """Detect potential fradulent transactions from the inputted transaction data."""
    data = state["transaction_data"]
    fraud_results = {}
    fraud_results.update(fraud_detection_llm.invoke({"data": data.get("data", [])}))
    state["fraud_data"] = fraud_results.get("data", [])

    return state

def risk_analysis_agent(state: AgentState) -> AgentState:
    """The risk analysis agent will go through each transaction and define a risk score."""
    data = {
        "data": state["fraud_data"]
    }

    risk_results = risk_classification_llm.invoke({"data": data.get("data", [])})
    state["risk_report"] = risk_results
    
    return state

def supervisor_agent(state: AgentState) -> AgentState:
    """Supervisor agent orchestrates the fraud analysis, transaction analysis, and risk report and exports it into a PDF compilance report and JSON dashboard data."""
    final_report = {
        "transactions": state["transaction_data"],
        "fraud_data": state["fraud_data"],
        "risk_report": state["risk_report"]
    }

    state["final_report"] = final_report
    
    return state


# initalize graph and its nodes
workflow = StateGraph(AgentState)
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("transaction_analysis", transaction_analysis_agent)
workflow.add_node("fraud_detector", fraud_detection_agent)
workflow.add_node("risk_analysis", risk_analysis_agent)  # will have the TA agent and the FD agent going into it
workflow.add_node("supervisor", supervisor_agent)

# build graph
workflow.add_edge("ingestion", "transaction_analysis")
workflow.add_edge("transaction_analysis", "fraud_detector")
workflow.add_edge("fraud_detector", "risk_analysis")
workflow.add_edge("transaction_analysis", "risk_analysis")  # parallel line
workflow.add_edge("risk_analysis", "supervisor")
workflow.set_entry_point("ingestion")

fraud_detector = workflow.compile()

# visualize the graph
# extra step, but this visualizes the graph we created in LangGraph
display(Image(fraud_detector.get_graph().draw_mermaid_png()))

## MAIN ##
if __name__ == "__main__":
    start = time.time()

    initial_state = {
        "transaction_data_sources": ["", "", ""],
        "transaction_data": {},
        "fraud_data": [],
        "risk_report": "",
        "final_report": ""
    }

    result = fraud_detector.invoke(initial_state)
    print(f"RAW DATA: {result["final_report"]}")
    print(result["final_report"].content)

    print(f"Process took {time.time() - start} seconds to run.")