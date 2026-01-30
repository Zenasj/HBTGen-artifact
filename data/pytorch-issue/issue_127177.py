import torch

for epoch in tqdm(range(1)):

    ore.train()

    tick_loss = 0
    tick = 0

    future_dataset = FutureDataset('/project/231/data/body_edited/deepdata/y240520240102d.csv')

    actions = []

    ore.reset_states()

    for idx in range(len(future_dataset)):
        
        train_data, label = future_dataset[idx]

        logits = ore(train_data.to("cuda:0").unsqueeze(0))
        final_action = torch.softmax(logits, dim=-1).argmax()

        if not actions:
            if final_action != 0:
                actions.append(final_action.item())
        elif final_action != actions[-1]:
            actions.append(final_action.item())

        loss = loss_function(logits.unsqueeze(0), label.to("cuda:0").type(torch.long).unsqueeze(0))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        tick_loss += loss.item()
        tick += 1

        if idx % 3000 == 0:
            print(f'当前已训练 {idx} 条 TICK')

    average_loss = tick_loss / tick

    print(f'重点操作：{actions}')
    print(f'平均损失：{average_loss}')

ore.eval()
ore.reset_states()
with torch.inference_mode():
    for idx in range(len(future_dataset)):
        
        train_data, label = future_dataset[idx]

        logits = ore(train_data.to("cuda:0").unsqueeze(0))

        print(torch.softmax(logits, dim=-1))