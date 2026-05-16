"""
Multi-Agent Supervisor Pattern - CrewAI Implementation

Uses CrewAI's hierarchical process with a manager agent that delegates
tasks to specialist agents automatically.
"""

from crewai import Agent, Crew, Process, Task

# --- Specialist Agents ---
underwriting_agent = Agent(
    role="Underwriting Specialist",
    goal="Assess risk accurately and provide competitive quotes",
    backstory=(
        "You are a senior underwriter with deep expertise in commercial property "
        "risk. You evaluate building construction, occupancy, protection class, "
        "and loss history to determine appropriate premiums."
    ),
    verbose=True,
    allow_delegation=False,
)

claims_agent = Agent(
    role="Claims Specialist",
    goal="Process claims efficiently while preventing fraud",
    backstory=(
        "You are an experienced claims adjuster who handles commercial property "
        "claims. You verify coverage, assess damages, and determine appropriate "
        "settlement amounts."
    ),
    verbose=True,
    allow_delegation=False,
)

policy_admin_agent = Agent(
    role="Policy Administration Specialist",
    goal="Maintain accurate policy records and process changes promptly",
    backstory=(
        "You are a policy administration expert who handles endorsements, "
        "renewals, cancellations, and certificate issuance. You ensure all "
        "changes comply with regulatory requirements."
    ),
    verbose=True,
    allow_delegation=False,
)

# --- Manager Agent (Supervisor) ---
manager_agent = Agent(
    role="Insurance Operations Manager",
    goal="Route customer requests to the right specialist and ensure quality responses",
    backstory=(
        "You are the operations manager overseeing underwriting, claims, and "
        "policy administration. You understand each team's capabilities and "
        "route work to the appropriate specialist."
    ),
    verbose=True,
    allow_delegation=True,
)

# --- Tasks ---
customer_request = Task(
    description=(
        "A customer has contacted us with the following request:\n\n"
        "'We need to add a newly acquired 30,000 sqft distribution center "
        "to our existing commercial property policy POL-2024-7891. The building "
        "is Class B construction with sprinklers. We also want to know if our "
        "recent water damage claim CLM-4401 will affect the premium.'\n\n"
        "Handle all aspects of this request by delegating to the appropriate "
        "specialists and synthesizing their responses."
    ),
    expected_output=(
        "A comprehensive response addressing: (1) the endorsement to add the "
        "new location, (2) the risk assessment for the new building, and "
        "(3) the impact of the claim on their premium."
    ),
    agent=manager_agent,
)


def main():
    crew = Crew(
        agents=[manager_agent, underwriting_agent, claims_agent, policy_admin_agent],
        tasks=[customer_request],
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=True,
    )
    result = crew.kickoff()
    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(f"{'='*60}")
    print(result)


if __name__ == "__main__":
    main()
