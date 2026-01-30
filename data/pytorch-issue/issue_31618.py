correct = 0.0
for label, batch in dataloader:
  pred = model(batch)
  correct += pred.eq(label.data.view_as(y_pred)).long().cpu().sum()
acc = correct / N
# acc is 0

correct = 0.0
for label, batch in dataloader:
  pred = model(batch)
  correct += pred.eq(label.data.view_as(y_pred)).long().cpu().sum()
acc = 100.0 * correct / N
# acc is an int, without the decimal