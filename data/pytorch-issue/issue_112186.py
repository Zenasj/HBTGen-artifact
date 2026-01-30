import torch.nn as nn

import torch
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")
    print(f"MPS device: {mps_device}")
    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 4

    print(y.to("cpu"))
    # Move your model to mps just like any other device
    model = torch.nn.ReLU()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
    print("MPS runs successfully")

trainer = pl.Trainer(
    max_epochs=45,
    accelerator='mps',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)