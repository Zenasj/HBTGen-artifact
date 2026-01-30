import torch

def apply_model(data_loader, model):
    for dataset_name_, audio_path_, reference_, x, xlen, y, ylen in data_loader:
        x, xlen, y, ylen = x.to(args.device), xlen.to(args.device), y.to(args.device), ylen.to(args.device)
        with torch.no_grad():
            log_probs, output_lengths, loss = map(model(x, xlen, y = y, ylen = ylen).get, ['log_probs', 'output_lengths', 'loss'])
        yield log_probs

@torch.no_grad()
def apply_model(data_loader, model):
    print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9)
    for dataset_name_, audio_path_, reference_, x, xlen, y, ylen in data_loader:
        x, xlen, y, ylen = x.to(args.device), xlen.to(args.device), y.to(args.device), ylen.to(args.device)
        log_probs, output_lengths, loss = map(model(x, xlen, y = y, ylen = ylen).get, ['log_probs', 'output_lengths', 'loss'])
        yield log_probs

def decorate_no_grad(*args, **kwargs):
    with torch.no_grad():
        return apply_model(*args, **kwargs)