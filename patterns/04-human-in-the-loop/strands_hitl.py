"""
Human-in-the-Loop Pattern - Strands Agents SDK Implementation

Implements approval gates using a custom tool that pauses execution
and requests human confirmation before proceeding with high-value actions.
"""

from strands import Agent, tool
from strands.models.bedrock import BedrockModel


# --- Approval Gate ---
def get_human_approval(action: str, details: dict) -> bool:
    """Request human approval for a high-stakes action."""
    print(f"\n{'='*60}")
    print("APPROVAL REQUIRED")
    print(f"{'='*60}")
    print(f"Action: {action}")
    print(f"Details: {details}")
    print(f"{'='*60}")

    while True:
        response = input("\nApprove this action? [yes/no]: ").strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        print("Please respond with 'yes' or 'no'.")


# --- Tools ---
@tool
def lookup_policy(policy_id: str) -> dict:
    """Look up policy details. This is a read-only, low-risk operation."""
    return {
        "policy_id": policy_id,
        "holder": "Acme Manufacturing",
        "premium": 85_000,
        "status": "active",
        "coverage_limit": 5_000_000,
    }


@tool
def process_refund(policy_id: str, amount: float, reason: str) -> dict:
    """Process a premium refund. HIGH-RISK: Requires human approval."""
    # Approval gate for financial actions
    approved = get_human_approval(
        action="Process Premium Refund",
        details={
            "policy_id": policy_id,
            "refund_amount": amount,
            "reason": reason,
        },
    )

    if not approved:
        return {"status": "rejected", "message": "Refund rejected by approver."}

    # Proceed with refund
    return {
        "status": "approved",
        "policy_id": policy_id,
        "refund_amount": amount,
        "confirmation_id": "REF-2024-8832",
    }


@tool
def cancel_policy(policy_id: str, effective_date: str, reason: str) -> dict:
    """Cancel an insurance policy. HIGH-RISK: Requires human approval."""
    approved = get_human_approval(
        action="Cancel Policy",
        details={
            "policy_id": policy_id,
            "effective_date": effective_date,
            "reason": reason,
        },
    )

    if not approved:
        return {"status": "rejected", "message": "Cancellation rejected by approver."}

    return {
        "status": "cancelled",
        "policy_id": policy_id,
        "effective_date": effective_date,
        "confirmation_id": "CXL-2024-1192",
    }


def main():
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514",
        region_name="us-west-2",
    )

    agent = Agent(
        model=model,
        tools=[lookup_policy, process_refund, cancel_policy],
        system_prompt=(
            "You are an insurance operations agent. You can look up policies freely, "
            "but refunds and cancellations require human approval (the tools will "
            "handle the approval flow). Always look up the policy first before "
            "taking any action. Explain what you intend to do before doing it."
        ),
    )

    response = agent(
        "Customer wants to cancel policy POL-2024-7891 effective immediately "
        "due to business closure. Process a pro-rata refund for the remaining term."
    )
    print(f"\nFinal Response: {response}")


if __name__ == "__main__":
    main()
