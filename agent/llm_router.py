from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Apply FedEx corporate cert for all HTTPS libs
cert_path = os.getenv("REQUESTS_CA_BUNDLE")
if cert_path:
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    os.environ["CURL_CA_BUNDLE"]     = cert_path
    os.environ["SSL_CERT_FILE"]      = cert_path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route_with_llm(question: str) -> dict:
    """
    Ask LLM which tools to use. Returns {"route": "stats" | "narrative" | "both", "reason": "..."}
    """
    system_prompt = """
You are a routing agent that decides which tools are needed to answer a user question.

You must choose ONLY ONE of the following routes:
- stats: if the question asks for structured stats, rankings, averages, numbers, per-game outputs
- narrative: if the question needs explanations, reasons why, background, or historical context
- both: if the question needs both numbers and context

Respond in this exact JSON format:
{ "route": "...", "reason": "..." }
""".strip()

    user_prompt = f"Question: {question}"

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo"
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    import json
    return json.loads(response.choices[0].message.content.strip())