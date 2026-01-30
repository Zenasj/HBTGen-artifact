import torch

for epoch in range(10):
    net.train()
    # Good training.
    for data in trainloader:
        inputs, labels = data['images'], data['masks']

        for idx in range(0, len(inputs), 7):
            optimizer.zero_grad()

            outputs = net(inputs[idx:idx + 7])
            loss = criterion(outputs, labels[idx:idx + 7])
            loss.backward()
            optimizer.step()

    # Bad validation.
    net.eval()
    test_loss = 0.0
    test_times = 0
    for data in testloader:
        # !!!!!!!!👇
        with torch.no_grad():
            inputs, labels = data['images'], data['masks']

            for idx in range(0, len(inputs), 7):
                # or put no_grad here, leaking still happens.
                outputs = net(inputs[idx:idx + 7])
                loss = criterion(outputs, labels[idx:idx + 7])
                test_loss += loss.item()
                test_times += 1
    test_loss /= test_times