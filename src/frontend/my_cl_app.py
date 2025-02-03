from typing import Any, Dict, List, Optional, cast

import chainlit as cl
import httpx

# Constants
FASTAPI_URL = "http://localhost:8000/"
HTTP_TIMEOUT = 120.0


async def fetch_response(endpoint: str, payload: Optional[Dict[str, Any]] = None) -> str:
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
    answer = await fetch_response(
        "query", {"question": message.content, "use_llm": cl.user_session.get("use_llm", False)}
    )
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
    cl.user_session.set("use_llm", False)
    await cl.Message(
        content="You can activate/deactivate the use of the LLm with this button:", actions=actions
    ).send()


@cl.action_callback("change_llm_state")
async def toggle_llm(action: cl.Action) -> None:
    """Toggles the LLM activation state."""
    use_llm = cl.user_session.get("use_llm")
    cl.user_session.set("use_llm", not use_llm)
    status_msg = "LLM activated" if not use_llm else "LLM deactivated"
    await cl.Message(content=status_msg).send()
