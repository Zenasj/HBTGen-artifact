import torch

torch.cuda.set_rng_state(stored_state)
retrieved_state = torch.cuda.get_rng_state()