opt = SGD(model.parameters(), lr=0.1)
dataloader = DataLoader(...)
if scheduler_name == 'cyclic':
    scheduler = CyclicLR(opt, base_lr=0.01, max_lr=0.1)
elif scheduler_name == 'step':
    scheduler = StepLR(opt, 30)
else:
    scheduler = ExponentialLR(opt, 0.01)

for epoch in range(100):
    for batch in dataloader:
        train_batch(...)
        # currently
        if isinstance(scheduler, CyclicLR): scheduler.step()
        # proposed
        # if scheduler.steps_on_batch: scheduler.step()
    validate(...)
    # currently
    if isinstance(scheduler, [StepLR, ExponentialLR]): scheduler.step()
    # proposed
    # if scheduler.steps_on_epoch: scheduler.step()

class ReduceLROnPlateau(object):
    ...
    @property
    def steps_on_batch(self):
        return False

    @property
    def steps_on_epoch(self):
        return True