class Optimizer:
    ...
    def __enter__(self):
        self.zero_grad()
    def __exit__(self):
        self.step()

optimizer.zero_grad()  # STEP 1
...
loss.backward()
...
optimizer.step()  # STEP 2

with optimizer:
    ...
    loss.backward()
    ...