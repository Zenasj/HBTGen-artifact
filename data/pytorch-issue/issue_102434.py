import torch

def wrap_model_using_fsdp(self):
        params_no_grad = [n for n, p in self._model.named_parameters() if not p.requires_grad]
        
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            
        if len(params_no_grad) > 0:
            print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
                
        dtype = torch.float16
        mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        device_id = int(os.environ['RANK']) % torch.cuda.device_count()
                
        def get_module_class_from_name(module, name):
            modules_children = list(module.children())
            if module.__class__.__name__ == name:
                return module.__class__
            elif len(modules_children) == 0:
                return
            else:
                for child_module in modules_children:
                    module_class = get_module_class_from_name(child_module, name)
                    if module_class is not None:
                        return module_class
                
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        transformer_cls_to_wrap = set()
        for layer_class in ['LlamaDecoderLayer']:
            transformer_cls = get_module_class_from_name(self._model, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
                
        self._wrapped_model = self._model = FSDP(
            self._model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=f'cuda:{device_id}'
        )

_init_dist_slurm(args.dist_backend)
args.world_size = int(os.environ['WORLD_SIZE'])
args.local_rank = int(os.environ['LOCAL_RANK'])
args.rank = int(os.environ['RANK'])
args.gpu = args.rank % torch.cuda.device_count()
print(
    "| distributed init (rank {}, world {}, local_rank {}): {}".format(
    args.rank, args.world_size, args.local_rank, args.dist_url
    ),
    flush=True,
)

def _init_dist_slurm(backend, port=None) -> None:
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    torch.distributed.init_process_group(backend=backend)

device_id

torch.profiler

multi-node

single-node

all_gather

matmul

ncclKernel_ReduceScatter_RING_LL_Sum_half

record_param_comms

ncclKernel_AllGather_RING_LL_Sum_int8_t

ShardingStrategy=HYBRID_SHARD

FULL_SHARD

def wrap_model_using_fsdp(self):
        params_no_grad = [n for n, p in self._model.named_parameters() if not p.requires_grad]
        
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            
        if len(params_no_grad) > 0:
            print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
                
        dtype = torch.float16
        mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
                
        def get_module_class_from_name(module, name):
            modules_children = list(module.children())
            if module.__class__.__name__ == name:
                return module.__class__
            elif len(modules_children) == 0:
                return
            else:
                for child_module in modules_children:
                    module_class = get_module_class_from_name(child_module, name)
                    if module_class is not None:
                        return module_class
                
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        transformer_cls_to_wrap = set()
        for layer_class in ['LlamaDecoderLayer']:
            transformer_cls = get_module_class_from_name(self._model, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
                
        self._wrapped_model = self._model = FSDP(
            self._model,
            #sharding_strategy=ShardingStrategy.FULL_SHARD,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            #sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            #device_id=f'cuda:{device_id}'
            device_id=torch.cuda.current_device()
        )

LLAVA

Huggingface

torch.profiler