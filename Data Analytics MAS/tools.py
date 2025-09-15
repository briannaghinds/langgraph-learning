import os
import pandas as pd
from langchain_core.tools import tool

@tool
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV, JSON, TXT, or XLSX into a Pandas Dataframe.
    Returns the loaded dataset, or None if an error occurs.
    
    Args
        path: string path to the dataset file
    """

    # ROBUST CHECK #1
    if not os.path.exists(path):
        print("Error opening file. Check path definition.")
        return None
    
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".json":
            return pd.read_json(path)
        elif ext == ".txt":
            return pd.read_csv(path, sep="\t", engine="python")
        elif ext == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        else:
            print("File extension not supported. Convert and try again.")
            return None
    except Exception as e:
        print(f"Error: File not found, {str(e)}.")
        return None    
    
@tool
def data_analysis(data: pd.DataFrame) -> dict:
    pass

@tool
def data_visualization(data: pd.DataFrame) -> str:
    pass