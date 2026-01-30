import torch

if int(re.search(r'\d+', torch.__version__).group()) >= 2:
    # for pytorch 2.0
    model =torch.compile(model)
    log.info(f"Compiled the model for speed up")

model.to(device)