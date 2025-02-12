from typing import Any, Dict, List, Optional, cast

import chainlit as cl
import httpx
from chainlit.input_widget import Select, Slider, Switch

# Constants
FASTAPI_URL = "http://localhost:8000/"
HTTP_TIMEOUT = 120.0


COMMANDS = [
    {
        "id": "Benchmark",
        "icon": "image",
        "description": "Benchmark a decoder on the corpus, choose between modern_bert_large_embed, "
        "modern_bert_base_embed, modern_bert_base, modern_bert_large, gte_base, gte_large, modern_bert_base_v2",
    },
]


async def fetch_response(
    endpoint: str, payload: Optional[Dict[str, Any]] = None, http_timeout: Optional[float] = None
) -> str:
    """Handles HTTP requests to the backend."""
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT if http_timeout is None else http_timeout
        ) as client:
            response = await client.post(FASTAPI_URL + endpoint, json=payload or {})
            if response.status_code == 200:
                return cast(str, response.json().get("answer", "No response available."))
            return f"Error: {response.status_code} - {response.text}"
    except httpx.RequestError:
        return "Backend is unavailable. Please try again later."


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handles incoming messages and calls FastAPI RAG API."""
    if message.command == "Benchmark":
        # User is using the Picture command
        await benchmark_encoder_picker()
    else:
        answer = await fetch_response(
            "query",
            {
                "question": message.content,
                "use_llm": cl.user_session.get("use_llm", False),
                "return_n_doc": cl.user_session.get("num_retrieve_docs", 4),
            },
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
    settings = await cl.ChatSettings(
        [
            Switch(id="use_llm", label="use LLM in answer", initial=False),
            Slider(
                id="num_retrieve_docs", label="Number of documents to retreive", initial=4, min=1
            ),
        ]
    ).send()
    await update_settings(settings)
    await cl.context.emitter.set_commands(COMMANDS)
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


@cl.on_settings_update
async def update_settings(settings: cl.ChatSettings) -> None:
    """Toggles the LLM activation state."""
    print("Setup agent with following settings: ", settings)
    use_llm = settings["use_llm"]
    if use_llm != cl.user_session.get("use_llm", False):
        cl.user_session.set("use_llm", use_llm)
    num_retrieve_docs = settings["num_retrieve_docs"]
    if num_retrieve_docs != cl.user_session.get("num_retrieve_docs", 4):
        cl.user_session.set("num_retrieve_docs", num_retrieve_docs)


async def benchmark_encoder_picker() -> None:
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(
                name="modern_bert_large_embed",
                payload={"value": "lightonai/modernbert-embed-large"},
                label="modern_bert_large_embed",
            ),
            cl.Action(
                name="modern_bert_base_embed",
                payload={"value": "nomic-ai/modernbert-embed-base"},
                label="modern_bert_base_embed",
            ),
            cl.Action(
                name="modern_bert_base",
                payload={"value": "answerdotai/ModernBERT-base"},
                label="modern_bert_base",
            ),
            cl.Action(
                name="modern_bert_large",
                payload={"value": "answerdotai/ModernBERT-large"},
                label="modern_bert_large",
            ),
            cl.Action(
                name="gte_base", payload={"value": "Alibaba-NLP/gte-base-en-v1.5"}, label="gte_base"
            ),
            cl.Action(
                name="gte_large",
                payload={"value": "Alibaba-NLP/gte-large-en-v1.5"},
                label="gte_large",
            ),
            cl.Action(
                name="modern_bert_base_v2",
                payload={"value": "Alibaba-NLP/gte-modernbert-base"},
                label="modern_bert_base_v2",
            ),
            cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
        ],
    ).send()

    if res and res.get("payload").get("value", "cancel") != "cancel":
        answer = await fetch_response(
            "benchmark_corpus",
            {
                "encoder_path": res.get("payload").get("value"),
            },
            http_timeout=500,
        )
        await cl.Message(
            content=answer,
        ).send()

    return
