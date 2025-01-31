# import torch
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
)


class LLMType(str, Enum):
    mistral_7b = "mistralai/Mistral-7B-Instruct-v0.3"


class LLMWrapper:
    def __init__(self, llm_type: LLMType = LLMType.mistral_7b) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_type.value, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_type.value)
        self.start_new_conversation()

    def start_new_conversation(self):
        self.past_key_values = DynamicCache()
        self.max_cache_length = self.past_key_values.get_max_cache_shape()
        self.messages = []

    def generate_answer(self, conversation, tools):
        # format and tokenize the tool use prompt
        self.messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tools=tools,
        ).to(self.model.device)
        if isinstance(self.past_key_values, SinkCache):
            inputs = {k: v[:, -self.max_cache_length :] for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=256, past_key_values=self.past_key_values
        )
        completion = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": completion})
        print(completion)
        return self.messages


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass


if __name__ == "__main__":
    tools = [get_current_weather]
    llm_wrapper = LLMWrapper()
    # format and tokenize the tool use prompt

    user_prompts = [
        ("Hello, I'm Thomas, what's your name?", None),
        ("Btw, yesterday I was on a rock concert.", None),
        ("What's my name?", None),
        ("What's the weather like in Paris?", None),
    ]
    user_prompts = [
        ("Hello, I'm Thomas, what's your name?", None),
        ("Est-ce que tu sais parler francais?", None),
        ("Quelles autres langues sais-tu parler?", None),
        ("What's my name?", None),
    ]
    messages = []
    for prompt, tools in user_prompts:
        messages = llm_wrapper.generate_answer(prompt, tools)

    print(messages)
    # meta-llama 3.2
    # gemma 2
    # phi 3.5
