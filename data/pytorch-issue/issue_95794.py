py
import torch

import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    texts = ["This is a text for the example."] * 16
    tokenized_texts = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")      
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    batch = tokenized_texts
    batch["input_ids"], batch["labels"] = data_collator.torch_mask_tokens(batch["input_ids"])

    model = torch.compile(model, backend="inductor")
    model.to("cuda")

    batch = {k: v.to("cuda") for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()


if __name__ == "__main__":
    main()