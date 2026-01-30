compile = dict(
    target='train_step',  # (train_step, forward, model)
    verbose=True,
    backend='aot_eager',  
    dynamic=False, 
)