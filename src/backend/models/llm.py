# import torch
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
)


class LLMType(str, Enum):
    mistral_7b = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class LLMResult:
    last_answer: str
    messages: List[Dict[str, str]]


class LLMWrapper:
    def __init__(self, llm_type: LLMType = LLMType.mistral_7b, device: str = "mps") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_type.value, torch_dtype=torch.bfloat16, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_type.value)
        self.start_new_conversation()
        self.device = device
        # self.model.to(self.device)

    def start_new_conversation(self) -> None:
        self.past_key_values = DynamicCache()
        self.max_cache_length = self.past_key_values.get_max_cache_shape()
        self.messages: List[Dict[str, str]] = []

    def generate_answer(self, prompt: str, tools: Optional[List[Any]] = None) -> LLMResult:
        # format and tokenize the tool use prompt
        self.messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tools=tools,
        ).to(self.device)
        if isinstance(self.past_key_values, SinkCache):
            inputs = {k: v[:, -self.max_cache_length :] for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs, max_new_tokens=256, past_key_values=self.past_key_values
        )
        completion = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": completion})
        # print(completion)
        return LLMResult(last_answer=completion, messages=self.messages)


if __name__ == "__main__":

    def get_historical_information(question: str) -> None:
        """
        Get historical information about a given question.

        Args:
            question: The question to get the answer to.
        """
        pass

    tools = [get_historical_information]
    # tools = [get_current_weather, get_current_time]
    # tools = [get_current_temperature, get_current_wind_speed]
    llm_wrapper = LLMWrapper()
    # format and tokenize the tool use prompt

    user_prompts = [
        ("Hello, I'm Thomas, what's your name?", tools),
        ("Btw, yesterday I was on a rock concert.", None),
        ("What's my name?", None),
        ("What's the weather like in Paris?", None),
    ]
    user_prompts = [
        ("Hello, I'm Thomas, what's your name?", tools),
        # ("Est-ce que tu sais parler francais?", tools),
        # ("Quelles autres langues sais-tu parler?", tools),
        # ("What's my name?", None),
        ("When is born Emannuel Macron?", tools),
    ]
    for prompt, tools in user_prompts:
        messages = llm_wrapper.generate_answer(prompt, tools)

    print(messages.last_answer)
    # meta-llama 3.2
    # gemma 2
    # phi 3.5
