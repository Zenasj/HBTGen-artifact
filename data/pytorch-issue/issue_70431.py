import torch

for i in range(2):
    # fails on the 2nd pass i.e when i = 1
    model = Model(INPUT_DIMS, 9)
    model.to('cuda')
    policy, value = model(randInput, deterministic)

    print(torch.cuda.memory_allocated())    # 4584986624
    del model
    print(torch.cuda.memory_allocated())    # 4584986624

for i in range(2):
    # passes both passes (that's 2 passes in a 3 word sentence lol)
    model = Model(INPUT_DIMS, 9)
    model.to('cuda')
    with torch.no_grad():
        policy, value = model(randInput, deterministic)

    print(torch.cuda.memory_allocated())    # 192300544
    del model
    print(torch.cuda.memory_allocated())    # 2401792