"""
Human-in-the-Loop Pattern - LangGraph Implementation

Uses LangGraph's interrupt_before mechanism to pause the graph
at a specific node and wait for human input before continuing.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State ---
class HITLState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_action: dict | None
    human_approved: bool | None


# --- Tools ---
@tool
def lookup_policy(policy_id: str) -> dict:
    """Look up policy details (read-only, no approval needed)."""
    return {
        "policy_id": policy_id,
        "holder": "Acme Manufacturing",
        "premium": 85_000,
        "status": "active",
    }


@tool
def cancel_policy(policy_id: str, effective_date: str) -> dict:
    """Cancel a policy (requires prior human approval)."""
    return {
        "status": "cancelled",
        "policy_id": policy_id,
        "effective_date": effective_date,
        "confirmation_id": "CXL-2024-1192",
    }


# --- Graph Nodes ---
tools = [lookup_policy, cancel_policy]
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

HIGH_RISK_TOOLS = {"cancel_policy"}


def agent_node(state: HITLState) -> dict:
    """LLM reasons and decides next action."""
    response = llm.invoke(
        [SystemMessage(content="You are an insurance operations agent.")]
        + state["messages"]
    )
    return {"messages": [response]}


def check_approval_needed(state: HITLState) -> str:
    """Route: if the tool call is high-risk, go to approval gate."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        tool_name = last.tool_calls[0]["name"]
        if tool_name in HIGH_RISK_TOOLS:
            return "approval_gate"
        return "tools"
    return END


def approval_gate(state: HITLState) -> dict:
    """
    This node is where the graph INTERRUPTS.
    The pending action is stored in state for human review.
    When resumed, human_approved will be set.
    """
    last = state["messages"][-1]
    action = last.tool_calls[0] if last.tool_calls else None
    return {"pending_action": action}


def post_approval(state: HITLState) -> str:
    """Route based on human's decision."""
    if state.get("human_approved"):
        return "tools"
    return "rejected"


def rejected_node(state: HITLState) -> dict:
    """Inform the agent that the action was rejected."""
    return {
        "messages": [HumanMessage(content="The action was REJECTED by the approver. Inform the user.")],
        "pending_action": None,
        "human_approved": None,
    }


# --- Graph Construction ---
graph = StateGraph(HITLState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("approval_gate", approval_gate)
graph.add_node("rejected", rejected_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", check_approval_needed, {
    "tools": "tools",
    "approval_gate": "approval_gate",
    END: END,
})
graph.add_conditional_edges("approval_gate", post_approval, {
    "tools": "tools",
    "rejected": "rejected",
})
graph.add_edge("tools", "agent")
graph.add_edge("rejected", "agent")

# Compile with checkpointer and interrupt
memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["approval_gate"])


def main():
    config = {"configurable": {"thread_id": "hitl-demo-1"}}

    # Start the conversation
    result = app.invoke(
        {"messages": [HumanMessage(content="Cancel policy POL-2024-7891 effective 2024-12-01.")]},
        config=config,
    )

    # Graph will pause at approval_gate -- simulate human review
    print("\n--- Graph paused at approval gate ---")
    print(f"Pending action: {result.get('pending_action')}")

    # Human approves and resumes
    human_decision = input("Approve? [yes/no]: ").strip().lower() == "yes"
    result = app.invoke(
        {"human_approved": human_decision},
        config=config,
    )
    print(f"\nFinal: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
