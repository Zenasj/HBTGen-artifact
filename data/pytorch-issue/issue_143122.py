import torch

py
for _ in range(2):
    outputs = x * torch.sum(params)
    loss = criterion(outputs, y)
    loss.backward(
        inputs=(params,),
        create_graph=True,
    )
    inner_optimizer.step()
    inner_optimizer.zero_grad()

meta_loss = loss  # type: ignore
# make_dot((meta_loss , lr)).view()
# input()
meta_loss.backward(inputs=(p,), create_graph=True)  # <<<----- This line!
outer_optimizer.step()  # type: ignore