compile = dict(
    target='train_step',  # (train_step, forward, model)
    verbose=False,
    backend='inductor',  
    dynamic=False, 
)