import torch
import transformers

if __name__ == "__main__":
    device = torch.device('cuda')
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/roberta-large-ner-english").to(device)

    model.eval()
    a = torch.randint(100, 2000, (128, 256), device=device)

    with torch.no_grad():
        out_not_compiled = model(input_ids=a, attention_mask=torch.ones_like(a)).logits

    model = torch.compile(model)
    with torch.no_grad():
        out_compiled = model(input_ids=a, attention_mask=torch.ones_like(a)).logits

    print(torch.sum(torch.abs(out_compiled - out_not_compiled)))  # tensor(0.4234, device='cuda:0')