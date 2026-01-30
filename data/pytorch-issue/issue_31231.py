for iter, batch in enumerate(dataset):
    #from dataset 1
    image = batch['image'].cuda()
    gt = batch['gt'].cuda()
    #from dataset 2
    target = batch['target'].cuda()
    target_gt = batch['target_gt'].cuda()
    search = batch['search'].cuda()
    search_gt = batch['search_gt'].cuda()
    optim.zero_grad()
    if iter % 3 == 0: # every 3 iters 1 time
      pred1, pred2, pred3 = model(image, image)
      loss = crit(pred3, gt)
      loss.backward() 
    else:
      pred1, pred2, pred3 = model(target, search)
      loss = crit(pred1, target_gt) + crit(pred2, search_gt) 
      loss.backward() 
    optim.step()