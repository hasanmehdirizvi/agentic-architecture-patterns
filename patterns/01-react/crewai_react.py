"""
ReAct Pattern - CrewAI Implementation

CrewAI wraps the ReAct loop inside its Agent abstraction. The agent
reasons about which tool to use, acts, observes, and repeats.
"""

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool


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


# Define the agent with its role and tools
claims_agent = Agent(
    role="Insurance Claims Analyst",
    goal="Accurately assess claim eligibility by checking policies and coverage",
    backstory=(
        "You are a senior claims analyst with 15 years of experience. "
        "You methodically verify policy details, check existing claims, "
        "and calculate remaining coverage before making recommendations."
    ),
    tools=[search_policies, check_claim_status, calculate_remaining_coverage],
    verbose=True,
)

# Define the task
assessment_task = Task(
    description=(
        "Customer CUST-001 wants to file a new claim for $500,000 in storm damage. "
        "Look up their policy, check existing claims, calculate remaining coverage, "
        "and determine if the new claim can be accommodated."
    ),
    expected_output=(
        "A clear assessment of whether the claim can be filed, including "
        "policy details, current claims, and remaining coverage calculations."
    ),
    agent=claims_agent,
)


def main():
    crew = Crew(
        agents=[claims_agent],
        tasks=[assessment_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
