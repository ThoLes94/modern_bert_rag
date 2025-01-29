import os

from transformers import AutoModelForMaskedLM, AutoTokenizer


class ModernBertHF:
    def __init__(self, model_id: str = "answerdotai/ModernBERT-base"):
        self.model_id = model_id
        self.load_model()

    def load_model(self):
        root_local_path = os.path.join(
            os.getcwd(),
            "models/",
        )
        tokenizer_path = os.path.join(
            root_local_path, "tokenizer/", self.model_id.replace("/", "-")
        )
        bert_path = os.path.join(
            root_local_path, "bert/", self.model_id.replace("/", "-")
        )
        if os.path.isdir(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.tokenizer.save_pretrained(tokenizer_path)

        if os.path.isdir(bert_path):
            self.model = AutoModelForMaskedLM.from_pretrained(bert_path)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
            self.model.save_pretrained(bert_path)

    def __call__(self, prompts: list[str]) -> os.Any:
        pass
