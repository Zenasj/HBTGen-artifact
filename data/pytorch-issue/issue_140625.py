import os

import torch
from torch.export._draft_export import draft_export
from vit_pytorch.na_vit import NaViT


def main():
    v = NaViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        token_dropout_prob=0.1,
    )

    v.eval()
    with torch.no_grad():

        imgs = [
            [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
            [torch.randn(3, 128, 256), torch.randn(3, 256, 128)],
            [torch.randn(3, 64, 256)],
        ]

        example_inputs = (imgs,)
        print("Running torch export...")
        draft_ep, _ = draft_export(v, example_inputs)

        print("Running AOT Compile...")
        aoti_model_path = torch._inductor.aoti_compile_and_package(
            draft_ep,
            example_inputs,
            package_path=os.path.join(os.getcwd(), "navit.pt2"),
        )


if __name__ == "__main__":
    main()