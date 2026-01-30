from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from torch.export import Dim
import torch

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device="cuda"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

def export_to_torchep(model, name, img_size=1024):
    "Save the model to pytorch ExportedProgram format."

    dummy_batch =  torch.randn(5, 3, img_size, img_size).to("cuda").type(torch.bfloat16)

    # dynamic shapes for model export
    batch_size = Dim("batch", min=2, max=20)
    #height = Dim("height", min=2, max=2048)
    #width = Dim("width", min=2, max=2048)
    dynamic_shapes = {
        "sample": {0: batch_size},
    }

    # Export the model to pytorch ExportedProgram format
    ep = torch.export.export(
        model.eval(),
        (dummy_batch,),
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )

    # Save the exported model
    torch.export.save(ep, f"checkpoints/compiled/{name}")
    print(
        f"Model exported to pytorch ExportedProgram format: checkpoints/compiled/{name}"  # noqa: E501
    )

    return ep

export_to_torchep(predictor.model.image_encoder.bfloat16(), "sam2.1_hiera_large_exported")