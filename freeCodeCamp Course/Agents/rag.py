"""NOTES
- a RAG agent is a Retrieval-Augmented Generation where the agent can reason, plan, and use different tools
- where it uses the prompt and context as input, feeds that through the LLM and then outputs an answer and can either loop or end

GRAPH OVERVIEW
Start -> LLM -> Retriever Agent -> LLM -> End
"""

# import libraries
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_chroma import chroma  # I WANT TO USE SQL INSTEAD
from langchain_core.tools import tool

# load llm
llm = ChatOllama(
    model="mistral"
    # model="gpt-oss:20b",
    temperature=0
)

# embedding model 
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# define pdf path
pdf_path = "Stock_Market_Performance_2024.pdf"

# load the pdf
pdf_loader = PyPDFLoader(pdf_path)

# check the pdf and start chunking process
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages.")
except Exception as e:
    print(f"~Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


