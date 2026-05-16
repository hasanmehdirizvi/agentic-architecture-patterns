"""
Multi-Agent Supervisor Pattern - LangGraph Implementation

Implements a supervisor node that routes to specialist agent nodes.
Each specialist is a sub-graph with its own tools and system prompt.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# --- State ---
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str


# --- LLMs ---
supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
specialist_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --- Supervisor Node ---
def supervisor_node(state: SupervisorState) -> dict:
    """Route the query to the appropriate specialist."""
    response = supervisor_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a routing supervisor. Based on the user's message, "
                    "decide which specialist to route to. Respond with ONLY one of: "
                    "underwriting, claims, policy_admin, FINISH\n\n"
                    "- underwriting: risk assessment, quotes, premium calculations\n"
                    "- claims: claim filing, status, adjudication\n"
                    "- policy_admin: policy changes, renewals, documents\n"
                    "- FINISH: if the query has been fully answered"
                )
            ),
        ]
        + state["messages"]
    )
    route = response.content.strip().lower()
    if route not in ("underwriting", "claims", "policy_admin"):
        route = "FINISH"
    return {"next_agent": route}


# --- Specialist Nodes ---
def underwriting_node(state: SupervisorState) -> dict:
    """Underwriting specialist handles risk and rating."""
    response = specialist_llm.invoke(
        [
            SystemMessage(
                content="You are an underwriting specialist. Assess risk and provide quotes."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def claims_node(state: SupervisorState) -> dict:
    """Claims specialist handles claim inquiries."""
    response = specialist_llm.invoke(
        [
            SystemMessage(
                content="You are a claims specialist. Handle claim status and adjudication."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def policy_admin_node(state: SupervisorState) -> dict:
    """Policy admin specialist handles policy changes."""
    response = specialist_llm.invoke(
        [
            SystemMessage(
                content="You are a policy admin specialist. Handle endorsements and renewals."
            ),
        ]
        + state["messages"]
    )
    return {"messages": [response]}


# --- Routing ---
def route_to_specialist(state: SupervisorState) -> str:
    """Route based on supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


# --- Graph ---
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("underwriting", underwriting_node)
graph.add_node("claims", claims_node)
graph.add_node("policy_admin", policy_admin_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_specialist,
    {"underwriting": "underwriting", "claims": "claims", "policy_admin": "policy_admin", END: END},
)
# After specialist responds, go back to supervisor for potential follow-up
graph.add_edge("underwriting", "supervisor")
graph.add_edge("claims", "supervisor")
graph.add_edge("policy_admin", "supervisor")

app = graph.compile()


def main():
    result = app.invoke(
        {"messages": [HumanMessage(content="I need a quote for a 50,000 sqft warehouse.")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
