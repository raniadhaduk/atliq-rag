"""
Step 0: Sanity check - can we talk to Groq at all?
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Did you create a .env file "
        "with GROQ_API_KEY=your_key_here in the project root?"
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "In one sentence, what is Retrieval Augmented Generation?"},
    ],
    temperature=0.2,
)

answer = response.choices[0].message.content

print("=" * 60)
print("Groq responded successfully!")
print("=" * 60)
print(answer)