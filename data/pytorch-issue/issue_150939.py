import torch
import torch.nn as nn

def interpolate_pos_encoding_new(
        self,
        embeddings: torch.Tensor,
        orig_img,
    ) -> torch.Tensor:
        """
        Adapted from hf transformers
        """

        num_positions = self.pos_embed.shape[1] - 1

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        dim = embeddings.shape[-1]
        patch_size = torch.tensor([14, 14]).to(torch.float32)
        orig_hw = torch.tensor(orig_img.shape[2:]).to(torch.float32)

        new_size = orig_hw // patch_size

        sqrt_num_positions = torch.tensor(num_positions**0.5).to(torch.int64)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        val = patch_pos_embed.to(torch.float32)
        out_size = torch.cat([torch.tensor([1, dim]), new_size]).to(torch.int64)
        if torch.onnx.is_in_onnx_export():
            patch_pos_embed = (
                torch.onnx.ops.symbolic(
                    "Resize",  # Uses onnx::Resize op
                    [val, torch.tensor([]), torch.tensor([]), out_size],
                    {},
                    dtype=val.dtype,
                    shape=out_size,
                    version=1,
                )
                .to(dtype=target_dtype)
                .to(orig_img.device)
            )
        else:
            patch_pos_embed = torch.nn.functional.interpolate(
                val,
                size=(int(new_size[0].item()), int(new_size[1].item())),
                mode="bicubic",
                antialias=False,
            ).to(dtype=target_dtype).to(orig_img.device)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)