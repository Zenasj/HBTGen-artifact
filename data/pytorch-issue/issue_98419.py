import torch
import torch.nn as nn

def add_optimizer_hooks(
    model,
    optimizers: Dict[torch.nn.Parameter, torch.optim.Optimizer],  # Per-parameter optimizers
):
    """Ugly FSDP analog to torch.distributed.optim._apply_optimizer_in_backward
    
    FSDP changes acc_grad every step, so we need to apply this before *each* `backward()`
    call, unlike the normal recipe where we only apply it once.
    """

    param_handles = torch.distributed.fsdp._traversal_utils._get_fsdp_handles(model)
    assert set(model.parameters()) == {i.flat_param for i in param_handles} == set(optimizers.keys())

    # We need to use the post backward stream so updates apply gradients are accumulated
    stream = torch.distributed.fsdp._common_utils._get_module_fsdp_state(model)._streams["post_backward"]

    for h in param_handles:
        # We're going to call this early, so if we don't override to a no-op FSDP proper will call it again and assert fail.
        h.prepare_gradient_for_optim = lambda: None

        p = h.flat_param
        assert hasattr(p, "_post_backward_hook_state")
        fsdp_acc_grad, _ = p._post_backward_hook_state

        def _opt_hook(optimizer, p, h, *_unused):
            assert p._post_backward_called

            with torch.cuda.stream(stream):
                # Use the class to get at `prepare_gradient_for_optim`
                h.__class__.prepare_gradient_for_optim(h)
                assert p.grad is not None
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # Cool that this is now the default
                assert p.grad is None

        fsdp_acc_grad.register_hook(functools.partial(_opt_hook, optimizers[p], p, h))