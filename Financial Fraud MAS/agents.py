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

# define llm
load_dotenv()
GOOGLE_API_KEY = os.getenv("G_API_TOKEN")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# bind the specific tools to each llm
ingestion_llm = llm.bind_tools([load_csv, fetch_api, query_database])
transaction_analysis_llm = llm.bind_tools([statistical_summary, time_series_analysis, outlier_detection])
fraud_detection_llm = llm.bind_tools([ml_fraud_model, rule_based_checker, graph_based_detection])
risk_classification_agent = llm.bind_tools([risk_scorer, regulatory_checker])
supervisor_llm = llm  # the supervisor has no tools

# define agent state
class AgentState(TypedDict):
    transaction_data_sources: list[str]  # there will be multiple sources
    transaction_data: dict
    fraud_data: list[str]
    risk_report: dict




# define agents


# initalize graph and its nodes


# build graph


## MAIN ##
if __name__ == "__main__":
    pass