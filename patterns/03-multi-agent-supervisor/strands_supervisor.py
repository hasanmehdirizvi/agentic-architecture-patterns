"""
Multi-Agent Supervisor Pattern - Strands Agents SDK Implementation

Uses agent-as-tool composition: specialist agents are wrapped as tools
that the supervisor agent can invoke based on the user's query.
"""

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514",
    region_name="us-west-2",
)


# --- Specialist Agent Definitions ---
@tool
def underwriting_agent(query: str) -> str:
    """Delegate to the underwriting specialist for risk assessment and rating questions."""
    agent = Agent(
        model=model,
        system_prompt=(
            "You are an underwriting specialist. You assess risk, calculate premiums, "
            "and make coverage recommendations. Answer concisely."
        ),
    )
    response = agent(query)
    return str(response)


@tool
def claims_agent(query: str) -> str:
    """Delegate to the claims specialist for claim status, filing, and adjudication."""
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a claims specialist. You handle claim inquiries, status checks, "
            "and adjudication decisions. Answer concisely."
        ),
    )
    response = agent(query)
    return str(response)


@tool
def policy_admin_agent(query: str) -> str:
    """Delegate to the policy admin specialist for policy changes and document requests."""
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a policy administration specialist. You handle endorsements, "
            "cancellations, renewals, and document generation. Answer concisely."
        ),
    )
    response = agent(query)
    return str(response)


def main():
    # Supervisor agent routes to specialists
    supervisor = Agent(
        model=model,
        tools=[underwriting_agent, claims_agent, policy_admin_agent],
        system_prompt=(
            "You are a supervisor agent for an insurance company. Your job is to "
            "understand the customer's request and delegate to the appropriate "
            "specialist agent. You can delegate to:\n"
            "- underwriting_agent: risk assessment, new quotes, premium calculations\n"
            "- claims_agent: claim filing, status, adjudication\n"
            "- policy_admin_agent: policy changes, renewals, documents\n\n"
            "Route the query to the right specialist, then synthesize their "
            "response for the customer."
        ),
    )

    # Example queries that route to different specialists
    queries = [
        "I need a quote for a new commercial property policy for a warehouse.",
        "What's the status of my claim CLM-4401?",
        "I need to add a new location to my existing policy POL-2024-7891.",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        response = supervisor(query)
        print(f"RESPONSE: {response}")


if __name__ == "__main__":
    main()
