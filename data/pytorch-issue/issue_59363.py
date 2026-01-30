import torch
import torch.nn as nn

if rank == 0:
    for batch in train_loader: 
        image = batch['image'].cuda(rank, non_blocking=True)
        prediction = swa_model(image)

def train(rank, num_epochs, world_size):
    init_process(rank, world_size)
    torch.manual_seed(0)
    
    model = create_model()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate * world_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    swa_start = 10
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)
    criteria = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    
    train_loader, val_loader = get_dataloader(rank, world_size)
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            with torch.cuda.amp.autocast(enabled=True):
                image = batch['image'].cuda(rank, non_blocking=True)
                mask = batch['mask'].cuda(rank, non_blocking=True)
                
                pred = model(image)
                
                loss = criteria(pred, mask.unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    if rank == 0:
        for batch in train_loader: 
            image = batch['image'].cuda(rank, non_blocking=True)
            prediction = swa_model(image)

if rank == 0:
        for batch in train_loader: 
            image = batch['image'].cuda(rank, non_blocking=True)
            prediction = swa_model(image)

if rank == 0:
        with torch.no_grad():
            for batch in train_loader: 
                image = batch['image'].cuda(rank, non_blocking=True)
                prediction = swa_model(image)

loss = losses.avg        
global_loss = global_meters_all_avg(rank, world_size, loss)         
if rank == 0:            
     print(f"Epoch: {epoch+1} Train Loss: {global_loss[0]:.3f}")

model = LeNet()
model.load_state_dict(torch.load("lenet_rank_0.pth"))

if rank == 0:
    torch.save(swa_model.module.state_dict(), f'lenet_{rank}.pth')

model = DistributedDataParallel(model, device_ids=[rank])    
swa_model = torch.optim.swa_utils.AveragedModel(model)

cnt += 1
if cnt == 10:
    loss = losses.avg        
    global_loss = global_meters_all_avg(rank, world_size, loss)         
    if rank == 0:            
        print(f"Epoch: {epoch+1} Train Loss: {global_loss[0]:.3f}")

for epoch in range(num_epochs):
        losses = AvgMeter()
        
        if rank == 0:
            print("Rank: {}/{} Epoch: [{}/{}]".format(rank, world_size, epoch+1, num_epochs))
        
        model.train()
        for images, labels in train_loader:
            with torch.cuda.amp.autocast(enabled=True):
                images = images.cuda(rank, non_blocking=True)
                labels = labels.cuda(rank, non_blocking=True)
                
                pred = model(images)
                loss = criterion(pred, labels)

            global_loss = loss.avg.detach() # get a new tensor shares the storage and does not require grad
            # assuming loss.avg is a single tensor, launch async comm
            # this allows this comm to overlap with backward computation 
            work = dist.all_reduce(global_loss, async_op=True) 

            losses.update(loss.cpu().item(), images.size(0))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        if rank == 0:
            work.wait() # make sure the allreduce is done
            print(f"Epoch: {epoch+1} Train Loss: {global_loss[0]:.3f}")

for epoch in range(num_epochs):
        losses = AvgMeter()
        
        if rank == 0:
            print("Rank: {}/{} Epoch: [{}/{}]".format(rank, world_size, epoch+1, num_epochs))
        
        model.train()
        for images, labels in train_loader:
            with torch.cuda.amp.autocast(enabled=True):
                images = images.cuda(rank, non_blocking=True)
                labels = labels.cuda(rank, non_blocking=True)
                
                pred = model(images)
                loss = criterion(pred, labels)

            losses.update(loss.cpu().detach().item(), images.size(0))
            global_loss = losses.avg  
            work = dist.all_reduce(global_loss, async_op=True)