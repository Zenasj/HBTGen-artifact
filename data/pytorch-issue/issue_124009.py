import torch

from random import randrange

x = X_test.reset_index(drop=True).iloc[randrange(len(X_test))]
example = torch.tensor([x]).to(torch.float32)

traced_model = torch.jit.trace(model, example)
traced_model.save("traced_energynn_model.pt")