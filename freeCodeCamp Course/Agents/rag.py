"""NOTES
- a RAG agent is a Retrieval-Augmented Generation where the agent can reason, plan, and use different tools
- where it uses the prompt and context as input, feeds that through the LLM and then outputs an answer and can either loop or end

GRAPH OVERVIEW
Start -> LLM -> Retriever Agent -> LLM -> End
"""

# import libraries

