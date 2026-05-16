"""
RAG Agent Pattern - LangGraph Implementation

Implements an agentic RAG flow where the agent can retrieve documents,
evaluate relevance, and reformulate queries -- all as explicit graph nodes.
"""

from typing import Annotated, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# --- State ---
class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[dict]
    needs_reformulation: bool
    attempt: int


# --- Setup Vector Store (simulated documents) ---
sample_docs = [
    Document(
        page_content=(
            "Commercial property policies cover sudden and accidental water damage "
            "from burst pipes, including during freeze events. The insured must "
            "demonstrate reasonable heating was maintained."
        ),
        metadata={"source": "policy-forms/CP-001.pdf", "section": "Coverage A"},
    ),
    Document(
        page_content=(
            "Exclusions: Gradual seepage, ground water, flood, sewer backup "
            "(unless endorsement CP-215 is attached). Maintenance-related failures "
            "are excluded if the insured failed to winterize."
        ),
        metadata={"source": "policy-forms/CP-001.pdf", "section": "Exclusions"},
    ),
    Document(
        page_content=(
            "Claims filing requirements: Notice within 48 hours of discovery. "
            "Insured must take reasonable steps to mitigate further damage. "
            "Detailed inventory of damaged property required within 30 days."
        ),
        metadata={"source": "claims-handbook/procedures.pdf", "section": "Filing"},
    ),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(sample_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --- Graph Nodes ---
def retrieve_node(state: RAGState) -> dict:
    """Retrieve documents based on current query."""
    docs = retriever.invoke(state["query"])
    doc_dicts = [
        {"content": d.page_content, "source": d.metadata.get("source", "unknown")}
        for d in docs
    ]
    return {"documents": doc_dicts}


def evaluate_node(state: RAGState) -> dict:
    """Evaluate if retrieved documents are sufficient to answer."""
    docs_text = "\n".join(d["content"] for d in state["documents"])
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Evaluate if these documents can answer the user's question. "
                    "Respond with ONLY 'sufficient' or 'insufficient'."
                )
            ),
            HumanMessage(
                content=f"Question: {state['query']}\n\nDocuments:\n{docs_text}"
            ),
        ]
    )
    needs_reform = "insufficient" in response.content.lower()
    return {"needs_reformulation": needs_reform}


def reformulate_node(state: RAGState) -> dict:
    """Reformulate the query for better retrieval."""
    response = llm.invoke(
        [
            SystemMessage(content="Rewrite this query to improve search results. Return ONLY the new query."),
            HumanMessage(content=f"Original: {state['query']}"),
        ]
    )
    return {"query": response.content.strip(), "attempt": state["attempt"] + 1}


def generate_node(state: RAGState) -> dict:
    """Generate a grounded answer from retrieved documents."""
    docs_text = "\n\n".join(
        f"[{d['source']}]: {d['content']}" for d in state["documents"]
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the question using ONLY the provided documents. "
                    "Cite sources in brackets. If the documents don't contain "
                    "the answer, say so explicitly."
                )
            ),
            HumanMessage(
                content=f"Question: {state['query']}\n\nDocuments:\n{docs_text}"
            ),
        ]
    )
    return {"messages": [response]}


# --- Routing ---
def should_reformulate(state: RAGState) -> str:
    """Decide whether to reformulate or generate."""
    if state["needs_reformulation"] and state["attempt"] < 2:
        return "reformulate"
    return "generate"


# --- Graph ---
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("reformulate", reformulate_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_conditional_edges("evaluate", should_reformulate, {
    "reformulate": "reformulate",
    "generate": "generate",
})
graph.add_edge("reformulate", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()


def main():
    result = app.invoke(
        {
            "query": "Does commercial property insurance cover burst pipe water damage in winter?",
            "messages": [],
            "documents": [],
            "needs_reformulation": False,
            "attempt": 0,
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
