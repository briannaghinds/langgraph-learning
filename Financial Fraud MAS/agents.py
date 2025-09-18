"""
@author: Brianna Hinds
Description: Agent definitions, graph initialization, and main run of the Financial Fraud MAS.
"""

# import libraries
import os
import json
import time
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
from tools import *
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage
from IPython.display import Image, display

# -----------------------------
# Load API Key and initialize LLM
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("G_API_TOKEN")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# Prompts
# -----------------------------
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
Return JSON array of flagged transactions.
"""

RISK_ANALYSIS_PROMPT = """
Assign risk scores to suspicious transactions and perform regulatory checks.
Return a combined 'risk_report' for all transactions.
"""

SUPERVISOR_PROMPT = """
Combine the results from all agents into a final risk report.
Export as JSON and prepare for PDF output if needed.
"""

# -----------------------------
# Bind LLM tools to agents
# -----------------------------
ingestion_llm = llm.bind_tools([load_csv, fetch_api, query_database], system_prompt=INGESTION_PROMPT, return_direct=True)
transaction_analysis_llm = llm.bind_tools([statistical_summary, time_series_analysis, outlier_detection], system_prompt=TRANSACTION_ANALYSIS_PROMPT, return_direct=True)
fraud_detection_llm = llm.bind_tools([ml_fraud_model, rule_based_checker, graph_based_detection], system_prompt=FRAUD_DETECTION_PROMPT, return_direct=True)
risk_classification_llm = llm.bind_tools([risk_scorer, regulatory_checker], system_prompt=RISK_ANALYSIS_PROMPT, return_direct=True)
supervisor_llm = llm  # supervisor does not use tools

# -----------------------------
# Agent State Definition
# -----------------------------
class AgentState(TypedDict):
    transaction_data_sources: Annotated[list[str], operator.add]
    transaction_data: Annotated[dict, operator.or_]
    fraud_data: Annotated[list[dict], operator.add]
    risk_report: Annotated[list[dict], operator.add]
    final_report: Annotated[dict, operator.or_]

# -----------------------------
# Helper: Robust LLM response parsing
# -----------------------------
def parse_llm_response(response) -> list:
    """Parse AIMessage or dict response robustly to return a list of dicts."""
    if isinstance(response, AIMessage):
        # First, try content
        if response.content.strip():
            try:
                parsed = json.loads(response.content)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                pass
        # fallback: function_call arguments
        func_args = response.additional_kwargs.get("function_call", {}).get("arguments")
        if func_args:
            try:
                parsed = json.loads(func_args)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                pass
    elif isinstance(response, dict):
        return response.get("data") or response.get("risk_report") or []
    return []

# -----------------------------
# Define Agents
# -----------------------------
def ingestion_agent(state: AgentState) -> AgentState:
    """Ingest transaction data from CSV, API, or DB sources."""
    all_data = []
    for source in state.get("transaction_data_sources", []):
        if source.endswith(".csv"):
            response = load_csv.invoke(source)
        elif source.startswith("http"):
            response = fetch_api.invoke(source)
        else:
            response = query_database.invoke(source, query="SELECT * FROM transactions")

        if "error" in response:
            print(f"Error loading {source}: {response['error']}")
        else:
            all_data.extend(response.get("data", []))

    state["transaction_data"] = {"data": all_data}
    return state

def transaction_analysis_agent(state: AgentState) -> AgentState:
    """Analyze transaction trends, anomalies, and patterns."""
    data = state.get("transaction_data", {})
    response = transaction_analysis_llm.invoke(
        f"Analyze this transaction dataset for statistics, time series, and outliers. Here is the data: {data}"
    )
    state["transaction_data"]["analysis"] = response
    return state

def fraud_detection_agent(state: AgentState) -> AgentState:
    """Detect potentially fraudulent transactions."""
    data = {"data": state.get("transaction_data", {}).get("data", [])}
    response = fraud_detection_llm.invoke(
        f"Run fraud detection and return ONLY JSON array of flagged transactions. Here is the data: {data}"
    )
    parsed = parse_llm_response(response)
    state["fraud_data"] = parsed
    return state

def risk_analysis_agent(state: AgentState) -> AgentState:
    """Assign risk scores and perform compliance checks."""
    fraud_data = state.get("fraud_data") or []
    data = {"data": fraud_data}
    response = risk_classification_llm.invoke(
        f"Analyze this transaction dataset, assign risk scores and perform compliance checks. Return a JSON array of objects. Here is the data: {data}"
    )
    parsed = parse_llm_response(response)
    if isinstance(parsed, list):
        state["risk_report"] = parsed
    elif isinstance(parsed, dict) and "risk_report" in parsed:
        state["risk_report"] = parsed["risk_report"]
    else:
        state["risk_report"] = []
    return state

def supervisor_agent(state: AgentState) -> AgentState:
    """Combine all results into a final report."""
    final_report = {
        "transactions": state.get("transaction_data", {}),
        "fraud_data": state.get("fraud_data", []),
        "risk_report": state.get("risk_report", [])
    }
    state["final_report"] = final_report
    return state

# -----------------------------
# Initialize Graph and Nodes
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("transaction_analysis", transaction_analysis_agent)
workflow.add_node("fraud_detector", fraud_detection_agent)
workflow.add_node("risk_analysis", risk_analysis_agent)
workflow.add_node("supervisor", supervisor_agent)

# build edges
workflow.add_edge("ingestion", "transaction_analysis")
workflow.add_edge("transaction_analysis", "fraud_detector")
workflow.add_edge("fraud_detector", "risk_analysis")
workflow.add_edge("transaction_analysis", "risk_analysis")  # parallel line
workflow.add_edge("risk_analysis", "supervisor")
workflow.set_entry_point("ingestion")

fraud_detector = workflow.compile()

# visualize graph
display(Image(fraud_detector.get_graph().draw_mermaid_png()))
png_bytes = fraud_detector.get_graph().draw_mermaid_png()
with open("fraud_detector_graph.png", "wb") as f:
    f.write(png_bytes)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    initial_state = {
        "transaction_data_sources": [
            "./Financial Fraud MAS/data/bank_transactions.csv",
            "./Financial Fraud MAS/data/cc_transactions.csv",
            "./Financial Fraud MAS/data/online_payments.csv"
        ],
        "transaction_data": {},
        "fraud_data": [],
        "risk_report": [],
        "final_report": {}
    }

    result = fraud_detector.invoke(initial_state)
    print("FINAL REPORT:", json.dumps(result.get("final_report", {}), indent=2))
    print(f"Process took {time.time() - start:.2f} seconds to run.")