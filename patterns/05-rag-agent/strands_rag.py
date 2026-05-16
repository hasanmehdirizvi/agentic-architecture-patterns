"""
RAG Agent Pattern - Strands Agents SDK Implementation

Uses Amazon Bedrock Knowledge Bases as the retrieval backend.
The agent decides when to search and can reformulate queries
for better retrieval results.
"""

import boto3
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

# Bedrock Agent Runtime client for Knowledge Base queries
bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime", region_name="us-west-2"
)

KNOWLEDGE_BASE_ID = "YOUR_KB_ID"  # Replace with your Bedrock KB ID


@tool
def search_knowledge_base(query: str, num_results: int = 5) -> dict:
    """
    Search the insurance policy knowledge base for relevant information.
    Use this when you need to look up policy terms, coverage details,
    exclusions, or regulatory requirements.
    """
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": num_results}
            },
        )
        results = []
        for item in response.get("retrievalResults", []):
            results.append(
                {
                    "content": item["content"]["text"],
                    "source": item.get("location", {}).get("s3Location", {}).get("uri", "unknown"),
                    "score": item.get("score", 0),
                }
            )
        return {"results": results, "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}


@tool
def search_claims_guidelines(query: str) -> dict:
    """
    Search claims handling guidelines and procedures.
    Use this for adjudication rules, approval thresholds, and process questions.
    """
    # In production, this would hit a second knowledge base or index
    # Simulated response for demonstration
    guidelines = {
        "water damage": {
            "content": (
                "Water damage claims require: (1) proof of sudden/accidental occurrence, "
                "(2) documentation within 48 hours, (3) mitigation efforts by insured. "
                "Excluded: gradual seepage, maintenance failures, flood (separate policy)."
            ),
            "source": "claims-handbook/water-damage-v3.pdf",
            "threshold": "Claims over $100,000 require senior adjuster review.",
        }
    }
    # Simple keyword matching for demo
    for key, value in guidelines.items():
        if key in query.lower():
            return {"results": [value], "query": query}
    return {"results": [], "query": query, "message": "No specific guidelines found."}


def main():
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514",
        region_name="us-west-2",
    )

    agent = Agent(
        model=model,
        tools=[search_knowledge_base, search_claims_guidelines],
        system_prompt=(
            "You are an insurance knowledge assistant with access to policy documents "
            "and claims guidelines. When answering questions:\n\n"
            "1. Search the knowledge base for relevant policy terms and conditions\n"
            "2. If the initial search doesn't yield good results, reformulate your query\n"
            "3. Always cite the source document when providing information\n"
            "4. Clearly distinguish between what the policy covers vs excludes\n"
            "5. If you cannot find the answer in the knowledge base, say so explicitly\n\n"
            "Never make up policy terms or coverage details -- only state what you "
            "can verify from the retrieved documents."
        ),
    )

    # Example: Agent will search KB, evaluate results, possibly reformulate
    response = agent(
        "A policyholder is asking whether their commercial property policy covers "
        "water damage from a burst pipe during a winter freeze. What are the "
        "coverage terms and claims filing requirements?"
    )
    print(response)


if __name__ == "__main__":
    main()
