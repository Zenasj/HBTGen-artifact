for i, data in enumerate(zip(self.optimizer.param_groups, values)):
    param_group, lr = data
    if isinstance(param_group["lr"], Tensor):
        lr_val = lr.item() if isinstance(lr, Tensor) else lr  # type: ignore[attr-defined]
        param_group["lr"].fill_(lr_val)
    else:
        param_group["lr"] = lr