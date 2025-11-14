"""
Getting Started Example

This is the simplest way to get started with Revenium OpenAI middleware.
Just import and start making requests!

Prerequisites:
    export OPENAI_API_KEY="your-openai-key"
    export REVENIUM_METERING_API_KEY="your-revenium-key"
"""

from openai import OpenAI
import revenium_middleware_openai

# Create client
client = OpenAI()

# Create metadata (optional - use what you need)
metadata = {
    "organization_id": "org-getting-started-demo",
    "product_id": "prod-getting-started",

    # Additional optional fields (uncomment to use):
    # "subscriber": {
    #     "id": "user-123",
    #     "email": "user@example.com",
    #     "credential": {
    #         "name": "api-key-name",
    #         "value": "api-key-value"
    #     }
    # },
    # "task_type": "chat-completion",
    # "trace_id": "session-abc-123",
    # "agent": "my-agent",
    # "subscription_id": "plan-pro",
    # "response_quality_score": 0.95
}

# Make request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello! Introduce yourself in one sentence."}
    ],
    usage_metadata=metadata
)

# Display response
print(response.choices[0].message.content)
print("\nUsage data sent to Revenium!")
