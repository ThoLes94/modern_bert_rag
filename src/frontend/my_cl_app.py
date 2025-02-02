from typing import List, cast

import chainlit as cl
import httpx

# Constants
FASTAPI_URL = "http://localhost:8000/"
HTTP_TIMEOUT = 120.0

# LLM Activation State
USE_LLM: List[bool] = [False]


async def fetch_response(endpoint: str, payload: dict = None) -> str:
    """Handles HTTP requests to the backend."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(FASTAPI_URL + endpoint, json=payload or {})
            if response.status_code == 200:
                return cast(str, response.json().get("answer", "No response available."))
            return f"Error: {response.status_code} - {response.text}"
    except httpx.RequestError:
        return "Backend is unavailable. Please try again later."


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handles incoming messages and calls FastAPI RAG API."""
    answer = await fetch_response("query", {"question": message.content})
    await cl.Message(content=answer).send()


@cl.on_chat_start
async def start() -> None:
    """Initializes the chat with an action button."""
    actions = [
        cl.Action(
            name="change_llm_state",
            icon="mouse-pointer-click",
            payload={},
            label="Activate/Deactivate the LLM!",
        )
    ]
    await cl.Message(content="Use the action button below:", actions=actions).send()


@cl.action_callback("change_llm_state")
async def toggle_llm(action: cl.Action) -> None:
    """Toggles the LLM activation state."""
    endpoint = "activate_llm" if not USE_LLM[0] else "deactivate_llm"
    response = await fetch_response(endpoint)

    if "Error" not in response and "Backend is unavailable. Please try again later." != response:
        USE_LLM[0] = not USE_LLM[0]  # Toggle state
        status_msg = "LLM activated" if USE_LLM[0] else "LLM deactivated"
        await cl.Message(content=status_msg).send()
    else:
        await cl.Message(content=response).send()
