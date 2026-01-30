import torch
import torchaudio.models as models

instance = models.Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=128,
    num_layers=4,
    depthwise_conv_kernel_size=31,
)
lengths = torch.randint(1, 400, (10,))  # (batch,)
input_ = torch.rand(10, int(lengths.max()), 80)  # (batch, num_frames, input_dim)
inputs = (input_, lengths)

# Failed to export
exported_program = torch.export.export(instance.eval(), inputs)
# exported_program = torch.export.export(instance.eval(), inputs, strict=False)
# exported_program = torch.export.export_for_training(instance.eval(), inputs)