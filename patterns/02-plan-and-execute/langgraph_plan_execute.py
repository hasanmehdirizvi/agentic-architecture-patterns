"""
Plan-and-Execute Pattern - LangGraph Implementation

Implements the pattern as a graph with explicit Planner and Executor nodes,
plus a Replanner that can revise the plan based on execution results.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State ---
class PlanExecuteState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    current_step: int
    step_results: list[str]
    final_answer: str


# --- Tools ---
@tool
def get_policy_details(policy_id: str) -> dict:
    """Retrieve full details of an insurance policy."""
    return {
        "policy_id": policy_id,
        "holder": "Acme Manufacturing",
        "coverages": {"property": 5_000_000, "liability": 2_000_000},
    }


@tool
def get_loss_history(policy_id: str) -> dict:
    """Get historical loss data for a policy."""
    return {"policy_id": policy_id, "loss_ratio": 0.15, "total_losses_3yr": 250_000}


@tool
def calculate_renewal_premium(policy_id: str, loss_ratio: float) -> dict:
    """Calculate renewal premium based on loss history."""
    base = 85_000
    return {"recommended_premium": int(base * (1.0 + loss_ratio * 0.5))}


# --- Nodes ---
tools = [get_policy_details, get_loss_history, calculate_renewal_premium]
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


def planner_node(state: PlanExecuteState) -> dict:
    """Decompose the task into steps."""
    response = planner_llm.invoke(
        [
            SystemMessage(
                content=(
                    "Break the user's request into 3-5 concrete steps. "
                    "Return each step on a new line, numbered."
                )
            ),
            state["messages"][0],
        ]
    )
    steps = [
        line.strip().lstrip("0123456789. ")
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    return {"plan": steps, "current_step": 0, "step_results": []}


def executor_node(state: PlanExecuteState) -> dict:
    """Execute the current step using tools."""
    step = state["plan"][state["current_step"]]
    context = "\n".join(state["step_results"]) if state["step_results"] else "None yet"
    msg = HumanMessage(
        content=f"Execute this step: {step}\nPrevious results:\n{context}"
    )
    response = executor_llm.invoke([msg])
    return {"messages": [msg, response]}


def should_use_tools(state: PlanExecuteState) -> str:
    """Check if executor wants to call tools."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "advance"


def advance_step(state: PlanExecuteState) -> dict:
    """Record result and move to next step."""
    last_msg = state["messages"][-1]
    result = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    results = state["step_results"] + [result]
    next_step = state["current_step"] + 1
    return {"step_results": results, "current_step": next_step}


def is_plan_complete(state: PlanExecuteState) -> str:
    """Check if all steps are done."""
    if state["current_step"] >= len(state["plan"]):
        return "summarize"
    return "executor"


def summarize_node(state: PlanExecuteState) -> dict:
    """Produce final answer from all step results."""
    summary = planner_llm.invoke(
        [
            SystemMessage(content="Summarize these results into a final recommendation."),
            HumanMessage(content="\n".join(state["step_results"])),
        ]
    )
    return {"final_answer": summary.content}


# --- Graph ---
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("advance", advance_step)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", is_plan_complete, {"executor": "executor", "summarize": "summarize"})
graph.add_conditional_edges("executor", should_use_tools, {"tools": "tools", "advance": "advance"})
graph.add_edge("tools", "executor")
graph.add_conditional_edges("advance", is_plan_complete, {"executor": "executor", "summarize": "summarize"})
graph.add_edge("summarize", END)

app = graph.compile()


def main():
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Prepare a renewal recommendation for policy POL-2024-7891."
                )
            ]
        }
    )
    print(result["final_answer"])


if __name__ == "__main__":
    main()
