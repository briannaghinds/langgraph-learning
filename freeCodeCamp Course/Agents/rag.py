"""NOTES
- a RAG agent is a Retrieval-Augmented Generation where the agent can reason, plan, and use different tools
- where it uses the prompt and context as input, feeds that through the LLM and then outputs an answer and can either loop or end

GRAPH OVERVIEW
Start -> LLM Agent -> Retriever Agent -> LLM Agent -> End
"""

# import libraries
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_chroma import chroma  # I WANT TO USE SQL INSTEAD
from langchain_core.tools import tool

# load llm
llm = ChatOllama(
    model="mistral",
    # model="gpt-oss:20b",
    temperature=0
)

# embedding model (this embedding model is compatable with the llm)
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# define pdf path
pdf_path = "./freeCodeCamp Course/Agents/Stock_Market_Performance_2024.pdf"

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
    chunk_size=1000,  # creating subsections of the text
    chunk_overlap=200  # keeps context with breaking up the text
)

# apply the text splitter to all the documents pages
pages_split = text_splitter.split_documents(pages)

# define where we will store the vector embeddings 
persist_dir = "./freeCodeCamp Course/Agents"
collection_name = "stock_market"

## DEBUG CHECK ##
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)
####

# create the database using embeddings model
try:
    vectorstore = FAISS.from_documents(
        documents=pages_split,
        embedding=embeddings
    )

    # save the index locally
    vectorstore.save_local("faiss_index")
    print("~Created FAISS vector store!")
except Exception as e:
    print(f"~Error setting up FAISS: {str(e)}")
    raise

# create the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}  # k = the amount of chunks to return 
)


# define agent tools
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document

    Args
        query: question the user wants answered
    """

    docs = retriever.invoke(query)

    # ROBUST CHECK
    if not docs: # answer doesn't exist
        return "I found no relevanty information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm = llm.bind_tools(tools)

# define the agent state and the condition edge check
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message conatins any tool calls."""
    result = state["messages"][1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {
    our_tool.name : our_tool for our_tool in tools
}

# define llm agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages  # feed the message in string format (system prompt + context)
    messages = llm.invoke(messages)

    return {"messages": [messages]}

# define retriever agent
def retriever_agent(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tools: {t['name']} with query {t['args'].get('query', 'No query provided.')}")

        # check if a valid tool is preset
        if not t["name"] in tools_dict:
            print(f"Tool: {t['name']} does not exist.")
            result = "Incorrect tool name, please retry and select a tool from the list of available tools."
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # append the ToolMessage
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
    
    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}

# initialize and build the graph and its nodes
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", retriever_agent)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()


# define a function to help run the agent workflow
def running_agent():
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)

## MAIN ##
if __name__ == "__main__":
    running_agent()
