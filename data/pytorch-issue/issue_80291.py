# when scheduler choose SequentialLR
lrs.SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[self.hparams.warm_up_iter])
...

# using pytorch_ligntning to train model, then return error because SequentialLR has no attribute 'optimizer'

model.fit()

self.optimizer = optimizer