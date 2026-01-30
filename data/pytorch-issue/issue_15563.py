import torch

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    model.train()
    for batch_idx, (image,label) in enumerate(loaders['train']):
        # move to GPU
        if use_cuda:
            image,label = image.cuda(), label.cuda()
        if (batch_idx +1) % 20 == 0:
            print('Batch Id '+ str(batch_idx +1))
        ## find the loss and update the model parameters accordingly
        ## record the average training loss, using something like
        ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - 
        ## train_loss))
        optimizer_scratch.zero_grad()
        output = model_scratch(image)
        loss = criterion_scratch(output,label)
        ##loss.backward(retain_graph=False)
        loss.backward()

        optimizer_scratch.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - 
        train_loss))
        torch.cuda.empty_cache() 
        ##gc.collect()