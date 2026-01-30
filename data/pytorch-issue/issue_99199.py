import torch

def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        
        # File: /home/titaiwang/pytorch/test/onnx/test_fx_to_onnx_with_onnxruntime.py:442, code: results.append(x[: x.size(0) - i, i : x.size(2), i:3])
        size = l_x_.size(0)
        sub = size - 0;  size = None
        size_1 = l_x_.size(2)
        getitem = l_x_[(slice(None, sub, None), slice(0, size_1, None), slice(0, 3, None))];  sub = size_1 = None
        size_2 = l_x_.size(0)
        sub_1 = size_2 - 1;  size_2 = None
        size_3 = l_x_.size(2)
        getitem_1 = l_x_[(slice(None, sub_1, None), slice(1, size_3, None), slice(1, 3, None))];  sub_1 = size_3 = None
        size_4 = l_x_.size(0)
        sub_2 = size_4 - 2;  size_4 = None
        size_5 = l_x_.size(2)
        getitem_2 = l_x_[(slice(None, sub_2, None), slice(2, size_5, None), slice(2, 3, None))];  sub_2 = size_5 = None
        size_6 = l_x_.size(0)
        sub_3 = size_6 - 3;  size_6 = None
        size_7 = l_x_.size(2)
        getitem_3 = l_x_[(slice(None, sub_3, None), slice(3, size_7, None), slice(3, 3, None))];  l_x_ = sub_3 = size_7 = None
        return (getitem, getitem_1, getitem_2, getitem_3)