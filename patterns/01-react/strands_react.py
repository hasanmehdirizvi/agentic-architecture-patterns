"""
ReAct Pattern - Strands Agents SDK Implementation

Demonstrates a reasoning + acting loop where the agent interleaves
thinking with tool calls until it reaches a final answer.
"""

from strands import Agent, tool
from strands.models.bedrock import BedrockModel


@tool
def search_policies(customer_id: str) -> dict:
    """Search insurance policies for a given customer."""
    # Simulated policy lookup
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
                "type": "water_damage",
            }
        ]
    }
    return {"claims": claims.get(policy_id, [])}


@tool
def calculate_remaining_coverage(policy_id: str) -> dict:
    """Calculate remaining coverage after pending/approved claims."""
    # Simplified calculation
    return {
        "policy_id": policy_id,
        "total_limit": 5_000_000,
        "claims_total": 150_000,
        "remaining_coverage": 4_850_000,
    }


def main():
    # Configure Bedrock model (Claude via AWS)
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514",
        region_name="us-west-2",
    )

    # Create agent with tools -- Strands handles the ReAct loop internally
    agent = Agent(
        model=model,
        tools=[search_policies, check_claim_status, calculate_remaining_coverage],
        system_prompt=(
            "You are an insurance claims assistant. Use the available tools to "
            "look up policy information, check claim statuses, and calculate "
            "remaining coverage. Reason step-by-step about what information "
            "you need before answering the customer's question."
        ),
    )

    # The agent will reason, call tools, observe results, and repeat
    response = agent(
        "Customer CUST-001 wants to know if they can file a new claim "
        "for $500,000 in storm damage. Check their policy and current claims."
    )
    print(response)


if __name__ == "__main__":
    main()
