import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dicts = []
state_dicts_names = []
for f in os.listdir("./models"):
  if f[-3:] == 'pth':
    print(f'Loading {f}')
    state_dicts_names.append(f)
    state_dicts.append(torch.load("./models/"+f, map_location=device))

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)