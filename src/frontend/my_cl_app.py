import chainlit as cl
import httpx

FASTAPI_URL = "http://localhost:8000/query"  # Adjust if needed


@cl.on_message  # type: ignore
async def main(message: cl.Message) -> None:
    """Handles incoming messages and calls FastAPI RAG API."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(FASTAPI_URL, json={"question": message.content})
            if response.status_code == 200:
                answer = response.json().get("answer", "Sorry, no response available.")
            else:
                answer = f"Error: {response.status_code} - {response.text}"
    except httpx.RequestError:
        answer = "Sorry, the backend is not available. Please try again later."

    await cl.Message(content=answer).send()
