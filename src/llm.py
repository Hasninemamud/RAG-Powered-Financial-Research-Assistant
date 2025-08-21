import requests
import os

HYPERBOLIC_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

def call_llm(question: str, snippets: list[dict]) -> str:
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Set HYPERBOLIC_API_KEY in your .env file.")

    # Build context from retrieved snippets
    context_text = "\n\n".join([f"(Page {s['page']}) {s['text']}" for s in snippets])
    prompt = f"""
You are a financial policy assistant.
Use the context below to answer the user's question.
Always cite page numbers when relevant.

Context:
{context_text}

Question: {question}
Answer:
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
    }

    payload = {
        # 👉 Try with a valid model name from Hyperbolic docs
        "model": "openai/gpt-oss-20b",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,  # some APIs require this
        "stream": False      # disable streaming unless supported
    }

    resp = requests.post(HYPERBOLIC_URL, headers=headers, json=payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Hyperbolic API error {resp.status_code}: {resp.text}") from e

    data = resp.json()
    return data["choices"][0]["message"]["content"]
