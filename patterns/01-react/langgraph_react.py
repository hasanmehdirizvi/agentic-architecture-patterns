"""
ReAct Pattern - LangGraph Implementation

Demonstrates the ReAct loop as an explicit graph: LLM node decides
whether to call tools or produce a final answer, with state tracking.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Tools ---
@tool
def search_policies(customer_id: str) -> dict:
    """Search insurance policies for a given customer."""
    policies = {
        "CUST-001": {
            "policy_id": "POL-2024-7891",
            "type": "Commercial Property",
            "coverage_limit": 5_000_000,
            "deductible": 25_000,
            "status": "active",
        }
    }
    return policies.get(customer_id, {"error": "Customer not found"})


@tool
def check_claim_status(policy_id: str) -> dict:
    """Check the status of claims against a policy."""
    claims = {
        "POL-2024-7891": [
            {
                "claim_id": "CLM-4401",
                "amount": 150_000,
                "status": "under_review",
                "filed_date": "2024-11-15",
            }
        ]
    }
    return {"claims": claims.get(policy_id, [])}


@tool
def calculate_remaining_coverage(policy_id: str) -> dict:
    """Calculate remaining coverage after pending/approved claims."""
    return {
        "policy_id": policy_id,
        "total_limit": 5_000_000,
        "claims_total": 150_000,
        "remaining_coverage": 4_850_000,
    }


# --- Graph Construction ---
tools = [search_policies, check_claim_status, calculate_remaining_coverage]
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """LLM decides: call a tool or produce final answer."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route based on whether the LLM wants to call tools."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # After tools, back to agent for reasoning

app = graph.compile()


def main():
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Customer CUST-001 wants to file a $500,000 storm damage claim. "
                        "Check their policy and current claims to see if they have enough coverage."
                    )
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
