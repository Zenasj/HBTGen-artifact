import torch
import torch.nn as nn

@contextlib.contextmanager
def autocast(whitelist_type=torch.float16, enabled=True):
    old_whitelist_type, old_status = torch.get_autocasting_state()
    torch.set_autocasting_state(whitelist_type, enabled)
    try:
        yield
    finally:
        torch.set_autocasting_state(original_whitelist_type, old_status)

with autocast():
    output = model(input)
    loss = loss_fn(output, target)
# The backward pass should be invoked outside the context manager.  All casting has been appropriately recorded as part of the forward pass.

def forward(self, x):
    x = self.layer_permitting_autocasting(x)
    with autocast(enabled=False):
        x = x.float()
        x = self.explicitly_float_layer(x)
    x = self.another_layer_permitting_autocasting(x)
    return x

scaler = torch.cuda.amp.AmpScaler()

for input, target in data:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

torch.autograd.backward(scaler.scale((output0, output1)), grad_tensors=(grad0, grad1))

torch.autograd.grad(scaler.scale((output0, output1)), model.parameters(), grad_outputs=(grad0, grad1))

scaler = AmpScaler()
...
for input, target in data:
    optimizer.zero_grad()
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

scaler = AmpScaler()
...
for input, target in data:
    optimizer.zero_grad()
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()

    # Gradients are scaled, so we clip to max_norm*scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm*scaler.get_scale())

    scaler.step(optimizer)
    scaler.update()

scaler = AmpScaler()
...
for input, target in data:
    optimizer.zero_grad()
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()

    scaler.unscale(optimizer)
    # Since the optimizer's owned gradients are unscaled, we can clip to max_norm directly:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    scaler.step(optimizer)
    scaler.update()

scaler = AmpScaler()
...
for input, target in data:
    optimizer.zero_grad()
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # We should scale outputs for the out-of-place backward pass
    grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

    # In general, the penalty term may depend nonlinearly on the out-of-place gradients, so to be safe,
    # manually unscale them before computing the penalty.  This unscale should be autograd-exposed.
    grad_params = [p*(1./scaler.get_scale()) for p in grad_params]

    # Compute the penalty term and add it to the loss.
    # The penalty term computation is effectively another snippet of forward pass, so it makes
    # sense to enable autocasting for this section as well:
    with autocast():
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

    # The usual scaling for backward will now accumulate leaf gradients that are appropriately scaled.
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

scaler = torch.cuda.amp.AmpScaler()
...
for input, target in data:
    optimizer0.zero_grad()
    optimizer1.zero_grad()
    with autocast():
        output0 = model0(input)
        output1 = model1(input)
        loss0 = loss_fn(2 * output0 + 3 * output1, target)
        loss1 = loss_fn(3 * output0 - 5 * output1, target)

    scaler.scale(loss0).backward(retain_graph=True)
    scaler.scale(loss1).backward()

    # Users can choose which optimizers receive explicit unscaling
    scaler.unscale(optimizer0)

    scaler.step(optimizer0)
    scaler.step(optimizer1)
    scaler.update()

scaler = AmpScaler()
...
for i, (input, target) in enumerate(data):
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
        loss = loss/iters_to_accumulate
    scaler.scale(loss).backward()
    if (i + 1) % iters_to_accumulate == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

scaler = AmpScaler(enabled=args.use_mixed_precision)
...
for input, target in data:
    optimizer.zero_grad()
    with autocast(enabled=args.use_mixed_precision):
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

scaler = AmpScaler()
...
for input, target in data:
    # Replay the batch, updating the scale if necessary, until we receive gradients that aren't inf/nan.
    while True:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.unscale(optimizer)
        if scaler._found_inf(optimizer).item():
            scaler.update()
        else:
            break
    scaler.step(optimizer)
    scaler.update()