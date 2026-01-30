import torch

def test_nanogpt(self):
        import sys

        sys.path.append("/home/titaiwang")

        from nanoGPT.model import GPT, GPTConfig

        # Load the model
        kwargs = {
            "block_size": 256,
            "vocab_size": 8096,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 128,
            "dropout": 0.0,
            "bias": False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        }
        config = GPTConfig(**kwargs)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True
        ):
            model = GPT(config)
        print("Done loading model")
        inputs = torch.arange(128).view(2, 64)
        targets = torch.arange(128).view(2, 64)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (inputs,),
            input_kwargs={
                "targets": targets,
            },
            verbose=True,
        )