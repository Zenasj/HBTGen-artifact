import torch
import torch.nn as nn

3
class TestTorchHSDP(DTensorTestBase):

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_torch_hsdp(self): 
        # NOTE: 
        # `TORCH_NCCL_AVOID_RECORD_STREAMS`x`use_deterministic_algorithms`=> grads have NaN
        #   0 x 0 => No
        #   0 x 1 => No
        #   1 x 0 => No
        #   1 x 1 => Yes
        
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        
        # mesh & fsdp2
        from torch.distributed.device_mesh import init_device_mesh # torch version: 2.4.1
        from torch.distributed._composable.fsdp import fully_shard, FSDPModule
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
        
        # llama model
        from transformers import AutoConfig, LlamaModel # transformer version: 4.46.1 (same for other version)
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = AutoConfig.from_pretrained(os.path.join(dir_path, "../llama/llama_config.json"))
        config.num_hidden_layers = 4
        config.hidden_size = 32
        config.intermediate_size = 88
        config.max_position_embeddings = 32
        config.vocab_size = 512

        torch.manual_seed(0)
        model: nn.Module = LlamaModel(config).cuda()

        # fsdp
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            # reshard_after_forward? # same NaN
            # mixed precision? # same NaN
        )
        for submod in model.modules():
            if isinstance(submod, LlamaDecoderLayer):
                fully_shard_fn(submod)
        fully_shard_fn(model)
        # model.set_reshard_after_backward()? # same NaN

        # data
        torch.manual_seed(self.rank)
        
        # microbatches
        for i in range(99):
            if self.rank == 0:
                print(f"[DEBUG] microbatch {i}")

            input = torch.randint(low=0, high=config.vocab_size, size=(4, 4), device="cuda")
            output = model(input).last_hidden_state
            output.mean().backward()

            # check NaN grad
            fsdp_params = []
            for module in cast(nn.Module, model).modules():
                if isinstance(module, FSDPModule):
                    if fsdp_param_group := module._get_fsdp_state()._fsdp_param_group:
                        fsdp_params += fsdp_param_group.fsdp_params
            for fsdp_param in fsdp_params:
                sharded_param = fsdp_param.sharded_param
                if not sharded_param.requires_grad:
                    continue
                if sharded_param.grad is None:
                    continue
                local_grad = sharded_param.grad._local_tensor
                self.assertEqual(torch.isnan(local_grad).sum().item(), 0, msg=f"{local_grad}")
                replicate_grad = sharded_param.grad.full_tensor()
                self.assertEqual(torch.isnan(replicate_grad).sum().item(), 0, msg=f"{replicate_grad}")