from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.model import ModernBertHF

if __name__ == "__main__":
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)

    text = "The capital of France is [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)
    outputs = model(**inputs)

    # To get predictions for the mask:
    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    print(masked_index)
    predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print("Predicted token:", predicted_token)
    bert = ModernBertHF()
