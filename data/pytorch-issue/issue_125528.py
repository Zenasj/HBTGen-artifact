import torch
import torch.distributed.checkpoint as dcp

def run(rank, world_size, resume="ckpt"):
    args = argparse.Namespace(local_rank=rank, world_size=world_size)
    state_dict = {"model": model, "optim": optim, "epoch": 0, "args":vars(args)}
    if os.path.exists(resume):
        dcp.load(state_dict, checkpoint_id=resume)
        _print(f"Resuming from epoch {state_dict['epoch']}")
    f = None
    for epoch in range(NUM_EPOCHS):
        state_dict['epoch'] = epoch
        try:
            #torch.manual_seed(epoch)
            x, y = _input()

            loss = loss_calc(model(x), y)

            _print(f"{epoch=} {loss=}")

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch>0 and epoch % SAVE_PERIOD == 0:
                if f is not None:
                    f.result()
                f = dcp.state_dict_saver.async_save(
                    state_dict, checkpoint_id=CHECKPOINT_DIR
                )

            if epoch>0 and FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                raise InjectedException("Fault injection!")
        except InjectedException as e:
            dist.barrier()

            _print("Trainer encountered exception:")
            traceback.print_tb(e.__traceback__)

            _print("Reloading model from last checkpoint!")
            if f is not None:
                f.result()

def dcp_load(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    no_dist=True,
):
    """
    Given a directory containing a DCP checkpoint, this function will load it into a
    state dict and return it as torch.load does.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    sd: STATE_DICT_TYPE = {}

    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=no_dist,
    )
    return sd