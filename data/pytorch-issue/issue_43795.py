import torch.nn as nn

import torch
import onnxruntime as rt

def _align_box(im_h, im_w, bbs,
               enlargement_factor, target_size, min_len, align='SHORTEST'):
    zero = torch.as_tensor(0.0).float().to(bbs.device)

    # enlarge the bounding box
    w = (bbs[:, 2] * enlargement_factor).ceil()
    h = (bbs[:, 3] * enlargement_factor).ceil()

    # CXCYWH to XYWH
    input_x = (bbs[:, 0] - w / 2).floor()
    cond = input_x < 0
    idx = torch.nonzero(cond).view(-1)
    # TODO: use scatter_add when fixed: https://github.com/pytorch/pytorch/issues/26472
    w = torch.scatter(w, 0, idx, w[idx] + input_x[idx])  # w[idx] += input_x[idx]
    input_x = torch.where(cond, zero, input_x)  # input_x[idx] = 0
    input_y = (bbs[:, 1] - h / 2).floor()
    cond = input_y < 0
    idx = torch.nonzero(cond).view(-1)
    # TODO: use scatter_add when fixed: https://github.com/pytorch/pytorch/issues/26472
    h = torch.scatter(h, 0, idx, h[idx] + input_y[idx])  # h[idx] += input_y[idx]
    input_y = torch.where(cond, zero, input_y)  # input_y[idx] = 0

    w = w.clamp(min=min_len)  # w[w < min_len] = min_len
    h = h.clamp(min=min_len)  # h[h < min_len] = min_len

    input_width = im_w - input_x
    input_width = torch.where(w < input_width, w, input_width)  # input_width[idx=w < input_width] = w[idx]
    input_height = im_h - input_y
    input_height = torch.where(h < input_height, h, input_height)  # input_height[idx=h < input_height] = h[idx]
    del w, h

    # If too far to the right shift the coordinates
    idx = input_width < min_len
    input_width = torch.where(idx, min_len, input_width)  # input_width[idx] = min_len
    input_x = torch.where(idx, im_w - min_len, input_x)  # input_x[idx] = im_w - min_len
    idx = input_height < min_len
    input_height = torch.where(idx, min_len, input_height)  # input_height[idx] = min_len
    input_y = torch.where(idx, im_h - min_len, input_y)  # input_y[idx] = im_h - min_len

    target_width = torch.full_like(input_width, target_size)
    target_height = torch.full_like(input_height, target_size)

    # align with constraint
    if align == 'SHORTEST':
        # Avoid `<=` because ONNX will export that as `Not` and `>` then we need another `Not`
        #  instead, first compute `>` and use `~` to be exported as `Not`
        nidx = input_width > input_height
        idx = ~nidx  # this requires patch_logical_not_opset9 until bitwise_not for bool is in PyTorch
        idx = torch.nonzero(idx).view(-1)
        target_height = torch.scatter(target_height, 0, idx, target_size * input_height[idx] / input_width[idx])

        idx = torch.nonzero(nidx).view(-1)
        target_width = torch.scatter(target_width, 0, idx, target_size * input_width[idx] / input_height[idx])
    elif align == 'LONGEST':
        idx = input_width > input_height
        nidx = ~idx  # this requires patch_logical_not_opset9 until bitwise_not for bool is in PyTorch
        idx = torch.nonzero(idx).view(-1)
        target_height = torch.scatter(target_height, 0, idx, target_size * input_height[idx] / input_width[idx])

        idx = torch.nonzero(nidx).view(-1)
        target_width = torch.scatter(target_width, 0, idx, target_size * input_width[idx] / input_height[idx])
    else:
        assert align == 'BOTH'

    return input_x, input_y, input_width, input_height, target_width, target_height

class Module(torch.nn.Module):
    def forward(self, img, bbs):
        _, _, height, width = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # When tracing (for onnx export this is Tensor)
        if not isinstance(height, torch.Tensor):
            height, width = torch.as_tensor(height).to(img.device), torch.as_tensor(width).to(img.device)

        input_x, input_y, input_width, input_height, target_width, target_height = _align_box(
            height.float(), width.float(), bbs, 1.5, 256, torch.as_tensor(3).float()
        )
        return input_x, input_y, input_width, input_height, target_width, target_height

model = Module()
img = torch.empty((1,3,256,512))
bbs = torch.as_tensor([[0, 0, 100, 100], [0, 0, 500, 100]]).float()

targets = ['input_x', 'input_y', 'input_width', 'input_height', 'target_width', 'target_height']
torch.onnx.export(model, (img, bbs), 'test.onnx',
                  verbose=True,
                  input_names=['img', 'bbs'],
                  output_names=targets,
                  opset_version=11,
                  dynamic_axes={'img': {2: 'h', 3: 'w'}, 'bbs': {0: 'rpn'}})

with onnx.export_dynamic():
    shape = t.shape