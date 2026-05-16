"""
Plan-and-Execute Pattern - Strands Agents SDK Implementation

Uses two agents: a Planner that decomposes the task into steps,
and an Executor that carries out each step with tools.
"""

import json

from strands import Agent, tool
from strands.models.bedrock import BedrockModel


# --- Tools for the Executor ---
@tool
def get_policy_details(policy_id: str) -> dict:
    """Retrieve full details of an insurance policy."""
    return {
        "policy_id": policy_id,
        "holder": "Acme Manufacturing",
        "type": "Commercial Package",
        "effective_date": "2024-01-01",
        "expiry_date": "2025-01-01",
        "coverages": {
            "property": 5_000_000,
            "liability": 2_000_000,
            "business_interruption": 1_000_000,
        },
    }


@tool
def get_loss_history(policy_id: str) -> dict:
    """Get historical loss data for a policy."""
    return {
        "policy_id": policy_id,
        "losses_3yr": [
            {"year": 2022, "amount": 50_000, "type": "water"},
            {"year": 2023, "amount": 200_000, "type": "fire"},
        ],
        "loss_ratio": 0.15,
    }


@tool
def calculate_renewal_premium(policy_id: str, loss_ratio: float) -> dict:
    """Calculate renewal premium based on loss history."""
    base_premium = 85_000
    adjustment = 1.0 + (loss_ratio * 0.5)
    return {
        "policy_id": policy_id,
        "base_premium": base_premium,
        "loss_adjustment": adjustment,
        "recommended_premium": int(base_premium * adjustment),
    }


def main():
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514",
        region_name="us-west-2",
    )

    # Planner agent: decomposes the task
    planner = Agent(
        model=model,
        system_prompt=(
            "You are a planning agent. Given a task, break it down into a numbered "
            "list of concrete steps. Output ONLY a JSON array of step descriptions. "
            "Do not execute anything -- just plan."
        ),
    )

    # Executor agent: carries out individual steps with tools
    executor = Agent(
        model=model,
        tools=[get_policy_details, get_loss_history, calculate_renewal_premium],
        system_prompt=(
            "You are an execution agent. You will be given a specific step to complete. "
            "Use your tools to accomplish that step and report the result concisely."
        ),
    )

    # Phase 1: Plan
    task = (
        "Prepare a renewal recommendation for policy POL-2024-7891. "
        "Need to review current coverage, loss history, and calculate new premium."
    )
    plan_response = planner(f"Create a plan for: {task}")
    print(f"=== PLAN ===\n{plan_response}\n")

    # Phase 2: Execute each step
    # In production, you would parse the JSON plan; here we demonstrate the pattern
    steps = [
        "Retrieve policy details for POL-2024-7891",
        "Get loss history for POL-2024-7891",
        "Calculate renewal premium based on the loss ratio",
    ]

    results = []
    for i, step in enumerate(steps, 1):
        print(f"=== EXECUTING STEP {i}: {step} ===")
        context = f"Previous results: {json.dumps(results)}" if results else ""
        result = executor(f"Execute this step: {step}\n{context}")
        results.append({"step": step, "result": str(result)})
        print(f"Result: {result}\n")

    print("=== EXECUTION COMPLETE ===")


if __name__ == "__main__":
    main()
