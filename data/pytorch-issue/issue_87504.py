import torch

device = torch.device("cuda")
model = AutoEncoder()
model.to(torch.device("cuda"))
model.load_state_dict(torch.load("model.pt"), map_location=device)

model.to(torch.device("cpu"))
torch.save(model.state_dict(), "model_cpu.pt")
discriminator.to(torch.device("cpu"))
torch.save(discriminator.state_dict(), "disc_cpu.pt")