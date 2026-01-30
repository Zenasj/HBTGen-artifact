import torch

'''
if torch.isinf(loss) or torch.isnan(loss):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt_bad.pt"))
    model_export(raw_model, os.path.join(out_dir, "model_bad.bin"), version=0)
    # Log gradients after each backward pass
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm():.4f}")

    err_msg = f"The logits are not stable: {logits}"
    assert not torch.logical_or(torch.isinf(logits), torch.isnan(logits)).any(), err_msg
'''