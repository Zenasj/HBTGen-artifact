def update_optimizer(epoch,optimizer,model,scheduler=None,new_lr=1e-4):

    print('Updating the optimiser...')
    new_params = [param for name, param in model.named_parameters() if any(layer in name for layer in model.unfreeze_scheduler[epoch]) and param.requires_grad]
    optimizer.add_param_group({'params': new_params, 'lr': new_lr})
    if scheduler is not None:
        scheduler.min_lrs = [0] * len(optimizer.param_groups)