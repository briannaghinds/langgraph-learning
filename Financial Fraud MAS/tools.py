"""
@author: Brianna Hinds
Description: Tool box defintions for Research MAS agent
"""

from langchain_core.tools import tool
import pandas as pd
import requests
import sqlite3
import os
import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb

## INGESTION AGENT TOOLS ##
# GOAL: loops until it has a complete dataset (transactions, users, metadata)
@tool
def load_csv(datapath: str) -> dict:
    """
    Loads a CSV file, normalizes column names, and returns a standardized dataset.
    
    Args:
        datapath (str): Path to the .csv file.
    
    Returns:
        dict: {
            "data": List of dict rows with normalized keys,
            "columns": Standardized column names that were detected,
            "original_columns": Original CSV column names
        }
    """

    COLUMN_MAPPINGS = {
        "amount": ["amount", "transaction_amount", "value", "amt"],
        "date": ["date", "transaction_date", "timestamp", "time"],
        "account_id": ["account_id", "acct_id", "user_id", "card_number"],
        "ip_address": ["ip_address", "ip", "source_ip"],
        "country": ["country", "location", "geo"]
    }

    # ROBUST CHECK 
    if not os.path.exists(datapath):
        return {"error": f"Define filepath: {datapath}, does not exist. Check definition and try again."}

    try:
        df = pd.read_csv(datapath)
        original_columns = df.columns.tolist()

        # normalize column names
        col_map = {}
        for standard_name, aliases in COLUMN_MAPPINGS.items():
            for alias in aliases:
                if alias in df.columns:
                    col_map[alias] = standard_name
                    break  # stop at first match

        df = df.rename(columns=col_map)

        return {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "original_columns": original_columns
        }

    except Exception as e:
        return {"error": f"Failed to load CSV: {str(e)}"}


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
    Generates descriptive statistics for financial transaction data.
    Provides high-level summary metrics (mean, median, standard deviation, etc.)
    for each numerical column in the dataset.

    Args:
        data (dict): A dictionary containing the transaction data.
                     Must include at least one numerical column (e.g., 'amount').

    Returns:
        dict: A dictionary with descriptive statistics:
              - "description": Pandas-style summary (count, mean, std, min, quartiles, max).
              - "<column_name>": Detailed stats for each numerical column (mean, median, std, sum, min, max).
              Returns {"error": "..."} if parsing or analysis fails.

    Example:
        >>> statistical_summary({"data": [{"amount": 120.0}, {"amount": 75.5}, {"amount": 200.0}]})
        {
            "results": {
                "description": {...},
                "amount": {
                    "mean": 131.83,
                    "median": 120.0,
                    "standard_deviation": 62.14,
                    "sum": 395.5,
                    "min": 75.5,
                    "max": 200.0
                }
            }
        }
    """

    # ROBUST CHECK
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])

        # statistical analysis
        stats = {}

        # define a description column
        stats["description"] = df.describe().to_dict()  # the output needs to be JSON-serializable to prevent errors
        for col in df.select_dtypes(include=["number"]):
            stats[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "standard_deviation": df[col].std(),
                "sum": df[col].sum(),
                "min": df[col].min(),
                "max": df[col].max()
            }

        return {"results": stats}
    
    except Exception as e:
        return {"error": f"Issue reading DataFrame object: {str(e)}"}


@tool
def time_series_analysis(data: dict) -> dict:
    """
    Performs time series analysis on financial transaction data.
    The function groups transaction amounts over different time windows (daily, weekly, monthly) 
    to uncover trends and patterns in spending or transfers.

    Args:
        data (dict): A dictionary containing the transaction data. 
                     Must include a 'date' column (transaction timestamp) 
                     and an 'amount' column (transaction value).

    Returns:
        dict: A dictionary with aggregated transaction totals:
              - "daily_transactions": Total transaction amounts per day.
              - "weekly_transactions": Total transaction amounts per week.
              - "monthly_transactions": Total transaction amounts per month.
              Returns {"error": "..."} if parsing or analysis fails.

    Example:
        >>> time_series_analysis({"data": [{"date": "2024-01-01", "amount": 120.0}, {"date": "2024-01-02", "amount": 75.5}]})
        {
            "results": {
                "daily_transactions": {"2024-01-01": 120.0, "2024-01-02": 75.5},
                "weekly_transactions": {"2023-12-31": 195.5},
                "monthly_transactions": {"2024-01-31": 195.5}
            }
        }
    """

    # ROBUST CHECK
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")

        stat = {
            "daily_transactions": df["amount"].resample("D").sum().to_dict(),
            "weekly_transactions": df["amount"].resample("W").sum().to_dict(),
            "monthly_transactions": df["amount"].resample("M").sum().to_dict()
        }

        return {"results": stat}

    except Exception as e:
        return {"error": f"Error opening the DataFrame: {str(e)}"}


@tool
def outlier_detection(data: dict) -> list:
    """
    Detects anomalous transactions in financial data.
    Flags transactions with unusually high or low amounts, and suspicious IP activity.

    Args:
        data (dict): A dictionary containing the transaction data.
                     Must include an 'amount' column and an 'ip_address' column.

    Returns:
        dict: A dictionary with detected anomalies:
              - "outliers": Transactions with anomalous amounts (based on IQR rule).
              - "ip_outlier": IP addresses with unusually high transaction counts.
              Returns {"error": "..."} if parsing or analysis fails.

    Example:
        >>> outlier_detection({"data": [{"amount": 5000, "ip_address": "192.168.1.10"}, {"amount": 50, "ip_address": "192.168.1.11"}]})
        {
            "results": {
                "outliers": [{"amount": 5000, "ip_address": "192.168.1.10"}],
                "ip_outlier": {"192.168.1.10": 1}
            }
        }
    """

    # ROBUST CHECK
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])
        outlier_results = {}

        # get the outliers via z-score and iqr
        q1 = df["amount"].quantile(0.25)
        q3 = df["amount"].quantile(0.75)
        iqr = q3-q1
        outliers = df[(df["amount"] < q1 - 1.5 * iqr) | df["amount"] > q3 + 1.5 * iqr]
        outlier_results["outliers"] = outliers.to_dict(orient="records")

        # get ip outliers
        ip_counts = df["ip_address"].value_counts()
        threshold = ip_counts.mean() + 2 * ip_counts.std()
        ip_outliers = ip_counts[ip_counts > threshold]
        outlier_results["ip_outlier"] = ip_outliers.to_dict()

        return {"results": outlier_results}

    except Exception as e:
        return {"error": f"Error defining data as a DataFrame: {str(e)}"}


## FRAUD DETECTION AGENT ##
# GOAL: detect potential fraud form the data and output a list of suspicious transactions
@tool
def ml_fraud_model(data: dict) -> dict:
    """
    Detects potential fraudulent transactions from input data.
    
    Args:
        data (dict): Dictionary containing transaction data. Expected format:
                     {"data": [ {feature1: val, feature2: val, ...}, ... ]}
    
    Returns:
        dict: Original data with an added "fraud_score" or "fraud_flag" for each transaction.
    """

    # ROBUST CHECK 
    if "error" in data:
        return data
    
    try:
        df = pd.DataFrame(data["data"])

        # call an ml model to detect transaction fraud (ensemble learning)
        iso = IsolationForest(contamination=0.01, random_state=42)  # using IsolationForest because the process will be unsupervised anomaly detection
        df["fraud_score"] = iso.fit_predict(df.select_dtypes(include=np.number))
        df["fraud_flag"] = df["fraud_score"].apply(lambda x: 1 if x == -1 else 0)

        result = df.to_dict(orient="records")
        return {"data": result}

    except Exception as e:
        return {"error": f"Error defining transaction data into DataFrame: {str(e)}"}
    

@tool
def rule_based_checker(data: dict) -> dict:
    """
    Flags suspicious transactions based on hard-coded rules:
      - Amount exceeds a threshold
      - Transactions from blacklisted accounts
      - Unusual transaction hours
    Returns a dictionary with flagged transactions.

    Args:
        data (dict): Dictionary with transaction data.

    Returns:
        dict: Contains "rule_flags" with flagged transaction details.
    """
    if "error" in data:
        return data

    try:
        df = pd.DataFrame(data["data"])
        df["rule_flag"] = 0

        # rule 1: high-value transactions
        df.loc[df["amount"] > 10000, "rule_flag"] = 1

        # rule 2: blacklisted accounts
        blacklist = ["12345", "99999"]
        df.loc[df["account_id"].isin(blacklist), "rule_flag"] = 1

        # rule 3: unusual hours (00:00 - 05:00)
        df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.hour
        df.loc[df["hour"].between(0, 5), "rule_flag"] = 1

        return {"rule_flags": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": f"Error in rule-based checking: {str(e)}"}


@tool
def graph_based_detection(data: dict) -> dict:  # finds rings/collusion
    """
    Detects collusion or ring-based fraud by analyzing connections between accounts.
    Constructs a simple graph from transactions (nodes=accounts, edges=shared metadata).
    Flags groups of accounts with unusual interconnections.

    Args:
        data (dict): Dictionary with transaction data.

    Returns:
        dict: Contains "graph_flags" with suspicious account clusters.
    """
    if "error" in data:
        return data

    try:
        import networkx as nx
        df = pd.DataFrame(data["data"])
        G = nx.Graph()

        # add edges between accounts that share metadata (IP, merchant)
        for _, row in df.iterrows():
            accounts = [row["account_id"]]
            meta = row.get("metadata", "")
            if meta:
                G.add_node(row["account_id"])
                # connect accounts sharing metadata
                shared = df[df["metadata"] == meta]["account_id"].tolist()
                for a in shared:
                    if a != row["account_id"]:
                        G.add_edge(row["account_id"], a)

        # find connected components (possible collusion rings)
        suspicious_clusters = [list(c) for c in nx.connected_components(G) if len(c) > 1]

        return {"graph_flags": suspicious_clusters}

    except Exception as e:
        return {"error": f"Error in graph-based detection: {str(e)}"}


## RISK ANALYSIS AGENT ##
# GOAL: for each suspicious transaction define a risk score and output a structured risk report (TransactionID -> Risk Score -> Reason) seperate each value by comma
@tool
def risk_scorer(data: dict) -> dict:
    """
    Assigns a rule-based risk score to transactions.
    
    Args:
        data (dict): Transaction dataset in the format:
                     {"data": [ { "amount": ..., "ip_address": ..., "account_id": ..., "date": ...}, ... ]}
    
    Returns:
        dict: Transactions with "risk_score" and "risk_reason".
    """
    if "error" in data:
        return data

    try:
        df = pd.DataFrame(data["data"])
        df["risk_score"] = 0
        df["risk_reason"] = ""

        for i, row in df.iterrows():
            score = 0
            reasons = []

            # Rule 1: High-value transaction
            if row.get("amount", 0) > 10000:
                score += 40
                reasons.append("High-value transaction")

            # Rule 2: Blacklisted account
            if row.get("account_id") in ["12345", "99999"]:
                score += 30
                reasons.append("Blacklisted account")

            # Rule 3: Unusual hours (00:00 - 05:00)
            if "date" in row and pd.notnull(row["date"]):
                hour = pd.to_datetime(row["date"], errors="coerce").hour
                if hour in range(0, 6):
                    score += 20
                    reasons.append("Unusual transaction time")

            # Rule 4: IP flagged (example suspicious IPs)
            if row.get("ip_address") in ["192.168.1.10", "10.0.0.99"]:
                score += 25
                reasons.append("Suspicious IP address")

            # Cap score at 100
            score = min(score, 100)

            df.at[i, "risk_score"] = score
            df.at[i, "risk_reason"] = ", ".join(reasons) if reasons else "Normal"

        return {"risk_report": df.to_dict(orient="records")}

    except Exception as e:
        return {"error": f"Error scoring transaction risk: {str(e)}"}


@tool
def regulatory_checker(data: dict) -> dict:
    """
    Checks transactions for potential regulatory compliance issues (AML/KYC).
    
    Args:
        data (dict): Transaction dataset in the format:
                     {"data": [ { "amount": ..., "account_id": ..., "country": ..., "date": ...}, ... ]}
    
    Returns:
        dict: Flags for regulatory concerns:
              - "aml_flags": High-value transactions above reporting threshold.
              - "structuring_flags": Accounts breaking large sums into smaller transactions.
              - "blacklist_flags": Accounts or countries on a watchlist.
    """
    if "error" in data:
        return data

    try:
        df = pd.DataFrame(data["data"])
        flags = {
            "aml_flags": [],
            "structuring_flags": [],
            "blacklist_flags": []
        }

        # Rule 1: AML threshold (e.g., > 10,000 must be reported)
        aml_threshold = 10000
        aml_cases = df[df["amount"] > aml_threshold]
        flags["aml_flags"] = aml_cases.to_dict(orient="records")

        # Rule 2: Structuring (multiple small transactions adding up > threshold in one day)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        daily_totals = df.groupby(["account_id", "date"])["amount"].sum().reset_index()
        structuring_cases = daily_totals[daily_totals["amount"] > aml_threshold]
        flags["structuring_flags"] = structuring_cases.to_dict(orient="records")

        # Rule 3: Blacklisted accounts/countries
        blacklist_accounts = ["12345", "99999"]
        blacklist_countries = ["IR", "KP"]
        blacklisted = df[(df["account_id"].isin(blacklist_accounts)) | 
                         (df.get("country", "").isin(blacklist_countries))]
        flags["blacklist_flags"] = blacklisted.to_dict(orient="records")

        return {"regulatory_report": flags}

    except Exception as e:
        return {"error": f"Error in regulatory check: {str(e)}"}


## SUPERVISOR AGENT TOOLS ##
# GOAL: combines the fraud and risk report and exports it into a PDF compilance report and JSON dashboard data