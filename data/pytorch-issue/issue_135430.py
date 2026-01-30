import torch.nn as nn

import torch._dynamo
import time
from torch.nn.attention import SDPBackend

torch._dynamo.reset()
compiled_model = get_model(config, "cuda")

compiled_model.decoder.model.setup_cache(
    1,
    config.decoder_args.max_seq_len,
    config.encoder_args.max_output_patches,
    device="cuda",
)
compiled_model.decoder.model = torch.compile(compiled_model.decoder.model, mode="max-autotune", fullgraph=True)  # type: ignore

encoder_outputs = (
    torch.randn(
        1, config.encoder_args.max_output_patches, config.encoder_args.output_dimensions
    )
    .to("cuda")
    .to(torch.bfloat16)
)
encoder_cache_pos = torch.arange(0, config.encoder_args.max_output_patches).to("cuda")


def run_test(n: int):
    print(f"===={n}====")
    for i in range(10):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
                    input_ids = torch.full(
                        (encoder_outputs.shape[0], n),
                        1,
                        dtype=torch.long,
                        device=encoder_outputs.device,
                    )
                    start = time.time()
                    cache_pos = torch.arange(0, n, device="cuda")
                    compiled_model.decoder.model(
                        input_ids=input_ids,
                        cache_pos=cache_pos,
                        encoder_outputs=encoder_outputs,
                        encoder_cache_pos=encoder_cache_pos,
                        use_encoder_cache=i != 0,
                    )
                    torch.cuda.synchronize()
                    print(time.time() - start)

run_test(1)
run_test(2)
run_test(3)
run_test(4)
run_test(5)