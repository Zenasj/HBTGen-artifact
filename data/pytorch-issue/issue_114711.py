import torch
import transformers
from torch._subclasses import fake_tensor
    
config = transformers.T5Config(
    vocab_size=8096, d_model=64, num_layers=2, num_heads=2
)
batch, seq = 4, 256

def create_args():
    return tuple()

def create_kwargs():
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    attention_mask = torch.ones((batch, seq), dtype=torch.bool)
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }

def create_model():
    return transformers.T5Model(config).eval()

fake_mode = fake_tensor.FakeTensorMode()

with fake_mode:
    fake_args = create_args()
    fake_kwargs = create_kwargs()
    fake_model = create_model()

    fake_model = torch.export.export(
        fake_model, args=fake_args, kwargs=fake_kwargs
    )

    # AssertionError: fake mode (<torch._subclasses.fake_tensor.FakeTensorMode object at 0x7fe1f6bf45d0>)
    #   from active fake mode 0 doesn't match mode
    #   (<torch._subclasses.fake_tensor.FakeTensorMode object at 0x7fe1e920fc90>) from fake tensor input 0
    model = fake_model.run_decompositions()