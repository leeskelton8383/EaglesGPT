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

def compose_final_answer(question: str, stats_result: str = None, narrative_context: str = None) -> str:
    """
    Compose a final answer using the question, stats result (if any), and narrative context (if any).
    """
    system_prompt = "You are a smart football assistant that synthesizes structured stats and historical context into clear, direct answers."

    parts = [f"Question: {question}"]

    import pandas as pd

    # Safely format stats_result (could be DataFrame or string)
    if isinstance(stats_result, pd.DataFrame) and not stats_result.empty:
        parts.append(f"Stats Result:\n{stats_result.to_string(index=False)}")
    elif isinstance(stats_result, str) and stats_result.strip():
        parts.append(f"Stats Result:\n{stats_result.strip()}")

    # Safely format narrative_context (should be string)
    if isinstance(narrative_context, str) and narrative_context.strip():
        parts.append(f"Narrative Context:\n{narrative_context.strip()}")

    user_prompt = "\n\n".join(parts)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()
