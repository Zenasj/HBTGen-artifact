import torch

def run(local_rank, world_size, param_size, num_params, work_dir):

    os.environ["RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    model = Model(param_size=param_size, num_params=num_params)
    model = DistributedDataParallel(model, gradient_as_bucket_view=True)
    _patch_model_state_dict(model)

    sz = sum(t.nelement() * t.element_size() for t in model.parameters())
    rank_0_print(f"Model size: {sz / 1_000_000_000.0} GB")
    rank_0_print("Saving the model with DCP...")

    checkpointer = _FileSystemCheckpointer(
        f"{args.work_dir}/dcp",
        sync_files=False,                                          
        single_file_per_rank=False,
        thread_count=1
    )

    begin_ts = time.monotonic()
    checkpointer.save(state_dict={"model": model})
    end_ts = time.monotonic()
    rank_0_print(f"Took {end_ts - begin_ts} seconds with DCP")