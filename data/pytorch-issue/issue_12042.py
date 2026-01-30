import torch

def save_checkpoint(epoch, model, best_top5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch+1, 'state_dict': model.state_dict(),
        'best_top5': best_top5, 'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filename)

if args.local_rank == 0:
    if is_best: save_checkpoint(epoch, model, best_top5, optimizer, is_best=True, filename='model_best.pth.tar')