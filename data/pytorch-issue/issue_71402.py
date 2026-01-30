import torch

def export_torchscript(model, im, model_name, optimize):
    # TorchScript model export
    try:
        logging.info(f'\nstarting export with torch {torch.__version__}...')
        f = Path('model.torchscript')

        ts = torch.jit.trace(model, (im, im), strict=False)
        print(ts(im, im))
        d = {"shape": im.shape, "names": model_name}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            # Export mobile interpreter version model (compatible with mobile interpreter)
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)

        logging.info(f'export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        logging.info(f'export failure: {e}')