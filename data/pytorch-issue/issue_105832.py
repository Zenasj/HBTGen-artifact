import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m').to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    model = torch.compile(model, backend="inductor")

    while True:
        x = tokenizer(['I like turtles.'], return_tensors='pt')
        x = x.to("cuda")
        output = model(**x)

if __name__ == '__main__':
    main()