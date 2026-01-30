for p in self.parameters():
            if p._base is not None and p._base.grad is not None:
                p._base._grad = None

def training_step(self, batch):
        self.zero_grad(set_to_none=True)
        for p in self.parameters():
            if p._base is not None and p._base.grad is not None:
                p._base._grad = None
        # rest loss compute logic
        ...

class ModelParallelStrategyWithCPUOffload(ModelParallelStrategy):
            def setup(self, trainer: "pl.Trainer") -> None:
                super().setup(trainer)
                # self.lightning_module.model.to("cpu")

                for layer in self.lightning_module.model.model.layers:
                    layer.input_layernorm.to("cpu")
                    layer.self_attn.q_proj.to("cpu")
                    layer.self_attn.k_proj.to("cpu")
                    layer.self_attn.v_proj.to("cpu")
                    layer.self_attn.o_proj.to("cpu")
                    layer.post_attention_layernorm.to("cpu")
                    layer.mlp.gate_proj.to("cpu")
                    layer.mlp.up_proj.to("cpu")
                    layer.mlp.down_proj.to("cpu")
                self.lightning_module.model.model.embed_tokens.to("cpu")
                self.lightning_module.model.model.norm.to("cpu")
                self.lightning_module.model.lm_head.to("cpu")