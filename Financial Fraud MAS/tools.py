"""
@author: Brianna Hinds
Description: Tool box defintions for Research MAS agent
"""

from langchain_core.tools import tool
import pandas as pd
import requests
import sqlite3
import os

## INGESTION AGENT TOOLS ##
# GOAL: loops until it has a complete dataset (transactions, users, metadata)
@tool
def load_csv(datapath: str) -> dict:
    """
    From a .csv define datapath, load the data into a Pandas DataFrame.
    Returns the loaded dataset, or None if an error occurs.

    Args
        datapath: datapath value of where the .csv is
    """
    
    # ROBUST CHECK 
    if not os.path.exists(datapath):
        return {"error": f"Define filepath: {datapath}, does not exist. Check definition and try again."}
    
    df = pd.read_csv(datapath)

    return {
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist()
    }

@tool
def fetch_api(url: str) -> dict:
    """
    Fetch data from a given API endpoint (expects JSON response).
    Returns a dictionary with data and metadata.
    
    Args:
        url: The API endpoint to fetch data from
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raise error for 4xx/5xx
        data = response.json()

        return {
            "data": data,
            "source": url,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "source": url,
            "status": "failed"
        }

@tool
def query_database(db_path: str, query: str) -> dict:
    """
    Run an SQL query on a SQLite database.
    Returns the result as a dictionary.
    
    Args:
        db_path: Path to SQLite database file (.db)
        query: SQL query string
    """
    if not os.path.exists(db_path):
        return {"error": f"Database file not found: {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        return {
            "data": [dict(zip(columns, row)) for row in rows],
            "columns": columns,
            "query": query,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "status": "failed"
        }

## TRANSACTION ANALYSIS TOOLS ##
# GOAL: parse the data for trends, anomalies, patterns and get a readable analysis report (tables, graphs, summaries)
@tool
def statistical_summary(data: dict) -> dict:
    """
    Used to define a statistical summary of the financial data, condensing the large dataset into a more manageable form.
    Returns a descriptive summary of the data.

    Args
        data: dictionary object of the data
    """

    # ROBUST CHECK
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])

        # statistical analysis
        stats = {}

        # define a description column
        stats["description"] = df.describe()
        for col in df.select_dtypes(include=["number"]):
            stats[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "standard_deviation": df[col].std(),
                "sum": df[col].sum(),
                "min": df[col].min(),
                "max": df[col].max()
            }

        return {"stats": stats}
    
    except Exception as e:
        return {"error": f"Issue reading DataFrame object: {str(e)}"}

@tool
def time_series_analysis(data: dict) -> dict:
    """"""

    # ROBUST CHECK
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])
    except Exception as e:
        return {"error": f"Error opening the DataFrame: {str(e)}"}

@tool
def outlier_detection(data: dict) -> dict:
    pass

## FRAUD DETECTION AGENT ##
# GOAL: detect potential fraud form the data and output a list of suspicious transactions
@tool
def ml_fraud_model(data: dict) -> dict:
    pass

@tool
def rule_based_checker(data: dict) -> dict:
    pass

@tool
def graph_based_detection(data: dict) -> dict:  # finds rings/collusion
    pass

## RISK ANALYSIS AGENT ##
# GOAL: for each suspicious transaction define a risk score and output a structured risk report (TransactionID -> Risk Score -> Reason) seperate each value by comma
@tool
def risk_scorer(transaction: str) -> dict:
    pass

@tool
def regulatory_checker():
    pass

## SUPERVISOR AGENT TOOLS ##
# GOAL: combines the fraud and risk report and exports it into a PDF compilance report and JSON dashboard data