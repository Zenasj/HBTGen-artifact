import torch
import transformers
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import os

with torch.no_grad():
    opt_config = transformers.OPTConfig(num_hidden_layers=2)
    opt_model = transformers.AutoModelForCausalLM.from_config(opt_config)
    opt_model.eval()
    assert torch.equal(
        opt_model.model.decoder.embed_tokens.weight, opt_model.lm_head.weight
    ), f"Expected lm_head weight is tied to the embed tokens weight"

    example_inputs = (torch.randint(0, opt_config.vocab_size, (2, 9), dtype=torch.int64),)
    float_out = opt_model(*example_inputs)

    if os.environ.get("NEW_EXPORT", "1") == "1":
        exported_model = torch.export.export_for_training(
            opt_model,
            args=example_inputs,
        ).module()
    else:
        exported_model = torch._export.capture_pre_autograd_graph(opt_model, args=example_inputs)
    exported_model.print_readable()
    quantizer = xiq.X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)