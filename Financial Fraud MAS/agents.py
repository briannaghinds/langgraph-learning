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
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage
from IPython.display import Image, display

# get API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("G_API_TOKEN")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# define system prompts per agent
INGESTION_PROMPT = """Ingest all sources from transaction_data_sources into transaction_data."""
TRANSACTION_ANALYSIS_PROMPT = """Analyze transactions for trends, anomalies, and outliers."""
FRAUD_DETECTION_PROMPT = """Detect fraud and return only JSON list of flagged transactions."""
RISK_ANALYSIS_PROMPT = """Assign risk scores and perform compliance checks. Return JSON list."""
SUPERVISOR_PROMPT = """Combine all results into a final structured risk report."""

# bind tools and prompts to each agent
ingestion_llm = llm.bind_tools([load_csv, fetch_api, query_database], system_prompt=INGESTION_PROMPT, return_direct=True)
transaction_analysis_llm = llm.bind_tools([statistical_summary, time_series_analysis, outlier_detection], system_prompt=TRANSACTION_ANALYSIS_PROMPT, return_direct=True)
fraud_detection_llm = llm.bind_tools([ml_fraud_model, rule_based_checker, graph_based_detection], system_prompt=FRAUD_DETECTION_PROMPT, return_direct=True)
risk_classification_llm = llm.bind_tools([risk_scorer, regulatory_checker], system_prompt=RISK_ANALYSIS_PROMPT, return_direct=True)
supervisor_llm = llm  # no tools needed

# define agent state
class AgentState(TypedDict):
    transaction_data_sources: Annotated[list[str], operator.add]
    transaction_data: Annotated[dict, operator.or_]              # raw + analysis
    fraud_data: Annotated[list[dict], operator.add]              # suspicious transactions
    risk_report: Annotated[list[dict], operator.add]             # scored suspicious transactions
    final_report: Annotated[dict, operator.or_]                  # final combined JSON

# parser for AI messagea
def parse_llm_response(response) -> list[dict]:
    if isinstance(response, AIMessage):
        if response.content.strip():
            try:
                parsed = json.loads(response.content)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                pass
        func_args = response.additional_kwargs.get("function_call", {}).get("arguments")
        if func_args:
            try:
                parsed = json.loads(func_args)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                pass
    elif isinstance(response, dict):
        # Flatten nested {"data": {"data": []}}
        if "data" in response:
            inner = response["data"]
            if isinstance(inner, dict) and "data" in inner:
                return inner["data"]
            return inner if isinstance(inner, list) else [inner]

    return []


## AGENTS ##
def ingestion_agent(state: AgentState) -> AgentState:
    """Load CSV/API/DB into transaction_data."""
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
    data = state.get("transaction_data", {})
    response = transaction_analysis_llm.invoke(
        f"Analyze this transaction dataset. Here is the data: {data}"
    )

    # Extract function_call args only, not the raw AIMessage
    if isinstance(response, AIMessage):
        func_args = response.additional_kwargs.get("function_call", {}).get("arguments")
        if func_args:
            try:
                state["transaction_data"]["analysis"] = json.loads(func_args)
            except:
                state["transaction_data"]["analysis"] = func_args
        else:
            state["transaction_data"]["analysis"] = response.content
    else:
        state["transaction_data"]["analysis"] = str(response)

    return state


def fraud_detection_agent(state: AgentState) -> AgentState:
    """Detect potentially fraudulent transactions using rules-based logic."""
    transactions = state.get("transaction_data", {}).get("data", [])
    fraud_data = []

    for txn in transactions:
        risk_score = 0
        reasons = []

        # threshold amount
        if txn.get("amount", 0) > 2000:
            risk_score += 0.7
            reasons.append("High amount transaction")

        # flag same day transactions
        duplicates = [
            t for t in transactions
            if t.get("account_id") == txn.get("account_id")
            and t.get("date") == txn.get("date")
            and t.get("amount") == txn.get("amount")
            and t.get("transaction_id") != txn.get("transaction_id")
        ]
        if duplicates:
            risk_score += 0.8
            reasons.append("Duplicate transaction same day")

        # flag unknown locations
        known_locations = {"AC001": ["New York"], "AC002": ["Chicago","Los Angeles"], 
                           "AC003": ["Miami","Lagos"], "AC004": ["New York"]}
        if txn.get("country") and txn.get("account_id") in known_locations:
            if txn["country"] not in known_locations[txn["account_id"]]:
                risk_score += 0.6
                reasons.append("Foreign location detected")

        if risk_score > 0:
            fraud_data.append({
                "transaction_id": txn.get("transaction_id"),
                "account_id": txn.get("account_id"),
                "amount": txn.get("amount"),
                "date": txn.get("date"),
                "risk_score": round(risk_score, 2),
                "reasons": reasons
            })

    state["fraud_data"] = fraud_data
    return state


def risk_analysis_agent(state: AgentState) -> AgentState:
    """Aggregate flagged transactions into per-account risk summary."""
    fraud_data = state.get("fraud_data", [])
    risk_report = {}

    # for each transaction in the fraud data
    for txn in fraud_data:
        acct = txn["account_id"]  # group by account_id
        if acct not in risk_report:  # count the amount of flagged transactions
            risk_report[acct] = {"total_flagged": 0, "transactions": []}
        risk_report[acct]["total_flagged"] += 1
        risk_report[acct]["transactions"].append(txn)

    # Convert dict to list for consistency with previous LLM output
    state["risk_report"] = [{"account_id": k, **v} for k,v in risk_report.items()]
    return state


def supervisor_agent(state: AgentState) -> AgentState:
    """Combine all results into final report with JSON and human-readable summary."""
    
    # clean the analysis of any duplicate data
    seen = set()
    clean_fraud = []
    for tx in state.get("fraud_data", []):
        tx_key = tx.get("transaction_id") or tx.get("payment_id") or f"{tx['account_id']}_{tx['date']}_{tx['amount']}"
        if tx_key not in seen:
            seen.add(tx_key)
            if "transaction_id" not in tx:
                tx["transaction_id"] = tx.get("payment_id", tx_key)
            clean_fraud.append(tx)
    
    state["fraud_data"] = clean_fraud

    # flatten the nested data keys
    transactions = state.get("transaction_data", {})
    if "analysis" in transactions:
        analysis = transactions["analysis"]
        while isinstance(analysis, dict) and "data" in analysis:
            analysis = analysis["data"]
        transactions["analysis"] = analysis
    state["transaction_data"] = transactions

    # build human readable summary from the agent's analysis
    summary_lines = []
    for account_summary in state.get("risk_report", []):
        account_id = account_summary.get("account_id", "Unknown")
        total = account_summary.get("total_flagged", 0)
        summary_lines.append(f"Account {account_id} has {total} flagged transactions:")
        for tx in account_summary.get("transactions", []):
            summary_lines.append(
                f"  - {tx['transaction_id']} | Amount: {tx['amount']} | Date: {tx['date']} | Risk: {tx['risk_score']} | Reasons: {', '.join(tx['reasons'])}"
            )

    # the final report contains 2 types: json (for dashboard) and normal text (for a financial analyst)
    state["final_report"] = {
        "json_report": {
            "transactions": state.get("transaction_data", {}),
            "fraud_data": state.get("fraud_data", []),
            "risk_report": state.get("risk_report", [])
        },
        "readable_report": "\n".join(summary_lines)
    }

    return state
####

# initialize and build graph
workflow = StateGraph(AgentState)
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("transaction_analysis", transaction_analysis_agent)
workflow.add_node("fraud_detector", fraud_detection_agent)
workflow.add_node("risk_analysis", risk_analysis_agent)
workflow.add_node("supervisor", supervisor_agent)

workflow.add_edge("ingestion", "transaction_analysis")
workflow.add_edge("transaction_analysis", "fraud_detector")
workflow.add_edge("fraud_detector", "risk_analysis")
workflow.add_edge("transaction_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "supervisor")
workflow.set_entry_point("ingestion")

fraud_detector = workflow.compile()

## GRAPH VISUALIZE ##
display(Image(fraud_detector.get_graph().draw_mermaid_png()))
with open("fraud_detector_graph.png", "wb") as f:
    f.write(fraud_detector.get_graph().draw_mermaid_png())
####

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
    print(json.dumps(result["final_report"], indent=2))
    print(f"Process took {time.time() - start:.2f} seconds")

    # adding the readable report into a txt file
    with open("./Financial Fraud MAS/financial_MAS.txt", "w") as f:
        f.write(result["final_report"]["readable_report"])
