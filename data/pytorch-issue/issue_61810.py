criterion = SomeLossFunc()
eps = 1e-6
loss = criterion(preds,targets)
if loss.isnan(): loss=eps
else: loss = loss.item()
loss = loss+ L1_loss + ...