import torch.nn as nn

import logging

import torch

import kornia
from kornia.utils import eye_like

torch._logging.set_logs(dynamo=logging.DEBUG)
torch._dynamo.config.verbose = True

align_corners = True
normalized_coordinates = True
device = torch.device("cpu")
dtype = torch.float32
batch_size = 3

# generate input data
height, width = 128, 64
eye_size = 3  # identity 3x3

patch_src = torch.rand(batch_size, 1, height, width, device=device, dtype=dtype)

# create base homography
dst_homo_src = eye_like(eye_size, patch_src)

# generate homography noise
homo_delta = torch.rand_like(dst_homo_src) * 0.3

dst_homo_src_i = dst_homo_src + homo_delta


op_optimized = torch.compile(kornia.geometry.transform.homography_warp, backend="inductor")

patch_dst_optimized = op_optimized(
    patch_src,
    dst_homo_src_i,
    (height, width),
    align_corners=align_corners,
    normalized_coordinates=normalized_coordinates,
)

# $TORCHDYNAMO_REPRO_AFTER="dynamo" python t.py
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = True

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_src_homo_dst_ : torch.Tensor, L_patch_src_ : torch.Tensor):
        l_src_homo_dst_ = L_src_homo_dst_
        l_patch_src_ = L_patch_src_
        xs = torch.linspace(0, 63, 64, device = device(type='cpu'), dtype = torch.float32)
        ys = torch.linspace(0, 127, 128, device = device(type='cpu'), dtype = torch.float32)
        truediv = xs / 63;  xs = None
        sub = truediv - 0.5;  truediv = None
        xs_1 = sub * 2;  sub = None
        truediv_1 = ys / 127;  ys = None
        sub_1 = truediv_1 - 0.5;  truediv_1 = None
        ys_1 = sub_1 * 2;  sub_1 = None
        meshgrid = torch.functional.meshgrid([xs_1, ys_1], indexing = 'ij');  xs_1 = ys_1 = None
        getitem = meshgrid[0]
        getitem_1 = meshgrid[1];  meshgrid = None
        base_grid = torch.stack((getitem, getitem_1), dim = -1);  getitem = getitem_1 = None
        permute = base_grid.permute(1, 0, 2);  base_grid = None
        grid = permute.unsqueeze(0);  permute = None
        grid_1 = grid.expand(3, -1, -1, -1);  grid = None
        src_homo_dst = l_src_homo_dst_.view(3, 1, 3, 3);  l_src_homo_dst_ = None
        to = grid_1.to(src_homo_dst);  grid_1 = None
        points_1 = to.reshape(-1, 64, 2);  to = None
        trans_1 = src_homo_dst.reshape(-1, 3, 3);  src_homo_dst = None
        trans_2 = torch.repeat_interleave(trans_1, repeats = 128, dim = 0);  trans_1 = None
        points_1_h = torch._C._nn.pad(points_1, [0, 1], 'constant', 1.0);  points_1 = None
        permute_1 = trans_2.permute(0, 2, 1);  trans_2 = None
        points_0_h = torch.bmm(points_1_h, permute_1);  points_1_h = permute_1 = None
        points_0_h_1 = torch.squeeze(points_0_h, dim = -1);  points_0_h = None
        z_vec = points_0_h_1[(Ellipsis, slice(-1, None, None))]
        abs_1 = torch.abs(z_vec)
        mask = abs_1 > 1e-08;  abs_1 = None
        add = z_vec + 1e-08
        truediv_2 = 1.0 / add;  add = None
        ones_like = torch.ones_like(z_vec);  z_vec = None
        scale = torch.where(mask, truediv_2, ones_like);  mask = truediv_2 = ones_like = None
        getitem_3 = points_0_h_1[(Ellipsis, slice(None, -1, None))];  points_0_h_1 = None
        points_0 = scale * getitem_3;  scale = getitem_3 = None
        flow = points_0.reshape([3, 128, 64, 2]);  points_0 = None
        warped_grid = flow.view(3, 128, 64, 2);  flow = None
        grid_sample = torch.nn.functional.grid_sample(l_patch_src_, warped_grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = True);  l_patch_src_ = warped_grid = None
        return (grid_sample,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('18c6cbf7f976b2076947af5f4dcb0618d1439075', 108)
    reader.tensor(buf0, (3, 3, 3), is_leaf=True)  # L_src_homo_dst_
    buf1 = reader.storage('f6ebd3c18a355ea3ca38dee4894ff94ed45abcdd', 98304)
    reader.tensor(buf1, (3, 1, 128, 64), is_leaf=True)  # L_patch_src_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/tmp/kornia/torch_compile_debug/run_2024_05_18_10_51_25_841050-pid_19688/minifier/checkpoints', autocast=False, backend='inductor')

# $TORCHDYNAMO_REPRO_AFTER="aot" python t.py
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = True





isolate_fails_code_str = None



# torch version: 2.3.0
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3060 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        iota = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        lt = torch.ops.aten.lt.Scalar(iota, 32.0)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 1.0);  convert_element_type = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        sub = torch.ops.aten.sub.Tensor(63, iota);  iota = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(63, mul_1);  mul_1 = None
        where = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        lt_1 = torch.ops.aten.lt.Scalar(iota_1, 64.0)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float32)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0);  convert_element_type_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(127, iota_1);  iota_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sub_2, torch.float32);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(127, mul_3);  mul_3 = None
        where_1 = torch.ops.aten.where.self(lt_1, add_1, sub_3);  lt_1 = add_1 = sub_3 = None
        div = torch.ops.aten.div.Tensor(where, 63);  where = None
        sub_4 = torch.ops.aten.sub.Tensor(div, 0.5);  div = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_4, 2);  sub_4 = None
        div_1 = torch.ops.aten.div.Tensor(where_1, 127);  where_1 = None
        sub_5 = torch.ops.aten.sub.Tensor(div_1, 0.5);  div_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_5, 2);  sub_5 = None
        view = torch.ops.aten.view.default(mul_4, [-1, 1]);  mul_4 = None
        expand = torch.ops.aten.expand.default(view, [64, 128]);  view = None
        view_1 = torch.ops.aten.view.default(mul_5, [1, -1]);  mul_5 = None
        expand_1 = torch.ops.aten.expand.default(view_1, [64, 128]);  view_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(expand, 2);  expand = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(expand_1, 2);  expand_1 = None
        cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], 2);  unsqueeze = unsqueeze_1 = None
        permute = torch.ops.aten.permute.default(cat, [1, 0, 2]);  cat = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_2, [3, -1, -1, -1]);  unsqueeze_2 = None
        view_2 = torch.ops.aten.view.default(arg0_1, [3, 1, 3, 3]);  arg0_1 = None
        clone = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_3 = torch.ops.aten.view.default(clone, [384, 64, 2]);  clone = None
        view_4 = torch.ops.aten.view.default(view_2, [-1, 3, 3]);  view_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [3, 128, 3, 3]);  unsqueeze_3 = None
        clone_1 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_5 = torch.ops.aten.view.default(clone_1, [384, 3, 3]);  clone_1 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(view_3, [0, 1], 1.0);  view_3 = None
        permute_1 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        bmm = torch.ops.aten.bmm.default(constant_pad_nd, permute_1);  constant_pad_nd = permute_1 = None
        squeeze = torch.ops.aten.squeeze.dim(bmm, -1);  bmm = None
        slice_1 = torch.ops.aten.slice.Tensor(squeeze, 2, -1, 9223372036854775807)
        abs_1 = torch.ops.aten.abs.default(slice_1)
        gt = torch.ops.aten.gt.Scalar(abs_1, 1e-08);  abs_1 = None
        add_2 = torch.ops.aten.add.Tensor(slice_1, 1e-08);  slice_1 = None
        reciprocal = torch.ops.aten.reciprocal.default(add_2);  add_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        full_default = torch.ops.aten.full.default([384, 64, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2 = torch.ops.aten.where.self(gt, mul_6, full_default);  gt = mul_6 = full_default = None
        slice_2 = torch.ops.aten.slice.Tensor(squeeze, 2, 0, -1);  squeeze = None
        mul_7 = torch.ops.aten.mul.Tensor(where_2, slice_2);  where_2 = slice_2 = None
        view_6 = torch.ops.aten.view.default(mul_7, [3, 128, 64, 2]);  mul_7 = None
        iota_2 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_8 = torch.ops.aten.view.default(iota_2, [3, 1, 1, 1]);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        full_default_1 = torch.ops.aten.full.default([1, 1, 1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select = torch.ops.aten.select.int(view_6, 3, 0)
        select_1 = torch.ops.aten.select.int(view_6, 3, 1);  view_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(select, 31.5);  select = None
        add_3 = torch.ops.aten.add.Tensor(mul_8, 31.5);  mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(select_1, 63.5);  select_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_9, 63.5);  mul_9 = None
        floor = torch.ops.aten.floor.default(add_3)
        floor_1 = torch.ops.aten.floor.default(add_4)
        add_5 = torch.ops.aten.add.Tensor(floor, 1)
        add_6 = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_6 = torch.ops.aten.sub.Tensor(add_5, add_3)
        sub_7 = torch.ops.aten.sub.Tensor(add_6, add_4)
        mul_10 = torch.ops.aten.mul.Tensor(sub_6, sub_7);  sub_6 = sub_7 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_3, floor)
        sub_9 = torch.ops.aten.sub.Tensor(add_6, add_4)
        mul_11 = torch.ops.aten.mul.Tensor(sub_8, sub_9);  sub_8 = sub_9 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_5, add_3)
        sub_11 = torch.ops.aten.sub.Tensor(add_4, floor_1)
        mul_12 = torch.ops.aten.mul.Tensor(sub_10, sub_11);  sub_10 = sub_11 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_3, floor);  add_3 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_4, floor_1);  add_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_12, sub_13);  sub_12 = sub_13 = None
        ge = torch.ops.aten.ge.Scalar(floor, 0)
        lt_2 = torch.ops.aten.lt.Scalar(floor, 64)
        ge_1 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_3 = torch.ops.aten.lt.Scalar(floor_1, 128)
        logical_and = torch.ops.aten.logical_and.default(ge_1, lt_3);  ge_1 = lt_3 = None
        logical_and_1 = torch.ops.aten.logical_and.default(lt_2, logical_and);  lt_2 = logical_and = None
        logical_and_2 = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_3 = torch.ops.aten.where.self(logical_and_2, convert_element_type_4, full_default_2);  convert_element_type_4 = full_default_2 = None
        view_10 = torch.ops.aten.view.default(where_3, [3, 1, 128, 64]);  where_3 = None
        full_default_3 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4 = torch.ops.aten.where.self(logical_and_2, convert_element_type_5, full_default_3);  convert_element_type_5 = full_default_3 = None
        view_11 = torch.ops.aten.view.default(where_4, [3, 1, 128, 64]);  where_4 = None
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_5 = torch.ops.aten.where.self(logical_and_2, mul_10, full_default_4);  logical_and_2 = mul_10 = full_default_4 = None
        view_12 = torch.ops.aten.view.default(where_5, [3, 1, 128, 64]);  where_5 = None
        index = torch.ops.aten.index.Tensor(arg1_1, [view_8, full_default_1, view_11, view_10]);  view_11 = view_10 = None
        mul_14 = torch.ops.aten.mul.Tensor(index, view_12);  index = view_12 = None
        ge_2 = torch.ops.aten.ge.Scalar(add_5, 0)
        lt_4 = torch.ops.aten.lt.Scalar(add_5, 64)
        ge_3 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_5 = torch.ops.aten.lt.Scalar(floor_1, 128)
        logical_and_3 = torch.ops.aten.logical_and.default(ge_3, lt_5);  ge_3 = lt_5 = None
        logical_and_4 = torch.ops.aten.logical_and.default(lt_4, logical_and_3);  lt_4 = logical_and_3 = None
        logical_and_5 = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_5, torch.int64)
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64);  floor_1 = None
        full_default_5 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6 = torch.ops.aten.where.self(logical_and_5, convert_element_type_6, full_default_5);  convert_element_type_6 = full_default_5 = None
        view_13 = torch.ops.aten.view.default(where_6, [3, 1, 128, 64]);  where_6 = None
        full_default_6 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_7 = torch.ops.aten.where.self(logical_and_5, convert_element_type_7, full_default_6);  convert_element_type_7 = full_default_6 = None
        view_14 = torch.ops.aten.view.default(where_7, [3, 1, 128, 64]);  where_7 = None
        full_default_7 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8 = torch.ops.aten.where.self(logical_and_5, mul_11, full_default_7);  logical_and_5 = mul_11 = full_default_7 = None
        view_15 = torch.ops.aten.view.default(where_8, [3, 1, 128, 64]);  where_8 = None
        index_1 = torch.ops.aten.index.Tensor(arg1_1, [view_8, full_default_1, view_14, view_13]);  view_14 = view_13 = None
        mul_15 = torch.ops.aten.mul.Tensor(index_1, view_15);  index_1 = view_15 = None
        add_7 = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        ge_4 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_6 = torch.ops.aten.lt.Scalar(floor, 64)
        ge_5 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_7 = torch.ops.aten.lt.Scalar(add_6, 128)
        logical_and_6 = torch.ops.aten.logical_and.default(ge_5, lt_7);  ge_5 = lt_7 = None
        logical_and_7 = torch.ops.aten.logical_and.default(lt_6, logical_and_6);  lt_6 = logical_and_6 = None
        logical_and_8 = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(floor, torch.int64);  floor = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        full_default_8 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_9 = torch.ops.aten.where.self(logical_and_8, convert_element_type_8, full_default_8);  convert_element_type_8 = full_default_8 = None
        view_16 = torch.ops.aten.view.default(where_9, [3, 1, 128, 64]);  where_9 = None
        full_default_9 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10 = torch.ops.aten.where.self(logical_and_8, convert_element_type_9, full_default_9);  convert_element_type_9 = full_default_9 = None
        view_17 = torch.ops.aten.view.default(where_10, [3, 1, 128, 64]);  where_10 = None
        full_default_10 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_11 = torch.ops.aten.where.self(logical_and_8, mul_12, full_default_10);  logical_and_8 = mul_12 = full_default_10 = None
        view_18 = torch.ops.aten.view.default(where_11, [3, 1, 128, 64]);  where_11 = None
        index_2 = torch.ops.aten.index.Tensor(arg1_1, [view_8, full_default_1, view_17, view_16]);  view_17 = view_16 = None
        mul_16 = torch.ops.aten.mul.Tensor(index_2, view_18);  index_2 = view_18 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, mul_16);  add_7 = mul_16 = None
        ge_6 = torch.ops.aten.ge.Scalar(add_5, 0)
        lt_8 = torch.ops.aten.lt.Scalar(add_5, 64)
        ge_7 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_9 = torch.ops.aten.lt.Scalar(add_6, 128)
        logical_and_9 = torch.ops.aten.logical_and.default(ge_7, lt_9);  ge_7 = lt_9 = None
        logical_and_10 = torch.ops.aten.logical_and.default(lt_8, logical_and_9);  lt_8 = logical_and_9 = None
        logical_and_11 = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_5, torch.int64);  add_5 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_6, torch.int64);  add_6 = None
        full_default_11 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12 = torch.ops.aten.where.self(logical_and_11, convert_element_type_10, full_default_11);  convert_element_type_10 = full_default_11 = None
        view_19 = torch.ops.aten.view.default(where_12, [3, 1, 128, 64]);  where_12 = None
        full_default_12 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_13 = torch.ops.aten.where.self(logical_and_11, convert_element_type_11, full_default_12);  convert_element_type_11 = full_default_12 = None
        view_20 = torch.ops.aten.view.default(where_13, [3, 1, 128, 64]);  where_13 = None
        full_default_13 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_14 = torch.ops.aten.where.self(logical_and_11, mul_13, full_default_13);  logical_and_11 = mul_13 = full_default_13 = None
        view_21 = torch.ops.aten.view.default(where_14, [3, 1, 128, 64]);  where_14 = None
        index_3 = torch.ops.aten.index.Tensor(arg1_1, [view_8, full_default_1, view_20, view_19]);  arg1_1 = view_8 = full_default_1 = view_20 = view_19 = None
        mul_17 = torch.ops.aten.mul.Tensor(index_3, view_21);  index_3 = view_21 = None
        add_9 = torch.ops.aten.add.Tensor(add_8, mul_17);  add_8 = mul_17 = None
        return (add_9,)
        
def load_args(reader):
    buf0 = reader.storage(None, 108)
    reader.tensor(buf0, (3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 98304)
    reader.tensor(buf1, (3, 1, 128, 64), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/tmp/kornia/torch_compile_debug/run_2024_05_18_10_57_22_980811-pid_20828/minifier/checkpoints', tracing_mode='real', check_str=None)