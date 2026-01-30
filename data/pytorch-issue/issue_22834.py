import torch

for trial in range(maxtrials):
    if inference:
        with torch.no_grad():
            ys = model(xs)
    else:
        optimizer.zero_grad()
        ys = model(xs)
        loss = criterion(ys, targets)
        if amp_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    finish = time.time()

    if finish-start >= mintime and trial >= mintrials:
        break