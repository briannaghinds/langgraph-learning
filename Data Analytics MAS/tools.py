import os
import pandas as pd
from langchain_core.tools import tool

@tool
def load_dataset(path: str) -> dict:
    """
    Load a dataset from CSV, JSON, TXT, or XLSX into a Pandas Dataframe.
    Returns the loaded dataset, or None if an error occurs.
    
    Args
        path: string path to the dataset file
    """

    # ROBUST CHECK
    if not os.path.exists(path):
        return {"error": f"File not found at path {path}"}
    
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".txt":
            df = pd.read_csv(path, sep="\t", engine="python")
        elif ext == ".xlsx":
            df = pd.read_excel(path, engine="openpyxl")
        else:
            return {"error": f"Unsupported file extension: {ext}"}
        
        return {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }
    
    except Exception as e:
        return {"error": str(e)}    
    
@tool
def data_analysis(data: dict) -> dict:
    """
    Perform simple descriptive statistics on numerical columns. 
    Returns a dictionary object of statistics per column and any important analysis information.

    Args
        data: dictionary object of the dataset {"data": [...], "columns": [...]}
    """

    # ROBUST CHECK 
    if "error" in data:
        return data  # if error exists just send it back
    
    try:
        df = pd.DataFrame(data["data"])
    except Exception as e:
        return {"error" : f"Error performing data analysis on dataset: {str(e)}"}

    pass

@tool
def data_visualization(data: dict) -> str:
    """
    Generate graphs/charts for numeric columns.
    Saves PNG(s) to ./graphs and returns a list of file paths to be called on by the report.

    Args
        data: dictionary object of the dataset
    """
    
    # ROBUST CHECK
    if "error" in data:
        return data