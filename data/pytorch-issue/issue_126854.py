for group in optimizer.param_groups:
    group.setdefault("initial_lr", copy.deepcopy(group["lr"]))