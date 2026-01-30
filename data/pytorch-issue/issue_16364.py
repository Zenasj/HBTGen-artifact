import torch
t = torch.tensor(0.)
optimizer = torch.optim.SGD([t],lr=.1)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=3)

print("---- First step, all good. ----")
loss = torch.Tensor([1.])
print(f"loss = {loss}")
print(f"id(loss) is {hex(id(loss))}.\nid(best) is {hex(id(lr_sched.best))}.")
lr_sched.step(loss)
best1 = lr_sched.best.item()
print(f"best1 = {best1}")

print("\n---- Second step, failure. ----")
loss += 1 #This is equivalent to lr_sched.best += 1, which is bad
print(f"loss = {loss}")
print(f"id(loss) is {hex(id(loss))}.\nid(best) is {hex(id(lr_sched.best))}.")
lr_sched.step(loss)
best2 = lr_sched.best.item()
print(f"best2 = {best2}")

print("\n---- Third step, mitigation ----")
loss = torch.Tensor([3.]) #New tensor
print(f"loss = {loss}")
print(f"id(loss) is {hex(id(loss))}.\nid(best) is {hex(id(lr_sched.best))}.")
lr_sched.step(loss)
best3 = lr_sched.best.item()
print(f"best3 = {best3}")

#Result check
assert best1 == best2 == best3 == 1., "On no, 'best' got worse :("

class ReduceLROnPlateau(object):
    #....
    def step(self, metrics, epoch=None):
        current = metrics
        #....
        if self.is_better(current, self.best):
            self.best = current.clone() ########## added `.clone()` #############
            self.num_bad_epochs = 0
        #....