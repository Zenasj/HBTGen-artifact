import torch.nn as nn

#filetypes = ('.cpp', '.cc', '.h', '.hpp')
filetypes = ('.cpp', '.cc')

#target_path = '..'
target_path = '../aten'

excluded_files = ['../c10/util/ConstexprCrc.h',
    '../aten/src/ATen/core/jit_type.h',
    '../aten/src/ATen/native/Math.h',
    '../c10/util/variant.h',
    '../c10/util/flags_use_no_gflags.cpp',
    '../caffe2/operators/cc_bmm_bg_op.h',
    '../aten/src/ATen/core/tensor_type.cpp',
    '../aten/src/ATen/native/Linear.cpp',
    '../aten/src/ATen/native/ConvolutionTBC.cpp',
    '../caffe2/share/fb/mask_rcnn/bbox_concat_batch_splits_op.h',
    '../aten/src/ATen/native/BatchLinearAlgebra.cpp',
    '../aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp',
    '../aten/src/ATen/native/cuda/DistributionTemplates.h',
    '../c10/util/sparse_bitset.h',
    '../torch/csrc/distributed/c10d/TCPStore.cpp',
    '../caffe2/fb/operators/calibration_op.h',
    '../torch/csrc/jit/testing/file_check.cpp',
    '../torch/csrc/jit/passes/concat_opt.cpp',
    '../torch/csrc/jit/tensorexpr/operators/reduction.cpp',
    '../torch/fb/operators/select_keys.cpp',
    '../torch/fb/operators/calibration/bucketize_calibration.cpp',
    '../fb/custom_ops/maskrcnn/int8/int8_aabb_roi_align.cpp',
    '../fb/custom_ops/maskrcnn/aabb/aabb_roi_align.cpp',
    '../caffe2/fb/tests/RecordIOHelper.cpp',
    '../test/cpp/api/rnn.cpp',
    '../torch/fb/training_toolkit/common/tdigest/tests/TestBufferedTDigest.cpp'
    ]

#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import re
import os

irange_header = "#include <c10/util/irange.h>"

# I recommend using https://regex101.com/ to understand this.
for_loop_regex = re.compile(
    r"for\s*\((?:int32_t|int64_t|uint32_t|int64_t|size_t|int|unsigned|auto|std::size_t|short|uint16_t|uint8_t) ([A-Za-z0-9_]+)\s*=\s*([^\s]+)\s*;\s*\1\s*<\s*([^\s]+)\s*;\s*(?:\+\+\1|\1\+\+)\s*\)\s*({?)")

header_regex = re.compile(r'#include ["<][^>"]+(?:[">])')

new_loop_zero = "for (const auto {loop_var} : c10::irange({upper_bound})){bracket}"
new_loop_range = (
    "for (const auto {loop_var} : c10::irange({lower_bound}, {upper_bound})){bracket}"
)

#header_insertion_points = (("c10", "alpha"), ("ATen/", "after"), ("torch/", "before"))

def find_c10(data : str) -> int:
    insert_at = -1
    for m in header_regex.finditer(data):
        if "c10/" in m.group(0):
            if insert_at is None:
                insert_at = m.span()[0]
            if irange_header > m.group(0):
                insert_at = m.span()[1]
    return insert_at

def find_ATen(data : str) -> int:
    insert_at = -1
    for m in header_regex.finditer(data):
        if "ATen/" in m.group(0):
            insert_at = m.span()[1]
    return insert_at

def find_torch(data : str) -> int:
    for m in header_regex.finditer(data):
        if "torch/" in m.group(0):
            return m.span()[0]
    return -1

def find_header_insertion_point(data: str) -> (int, str):
    """Look through headers to find an insertion point."""

    m = find_c10(data)
    if m != -1:
        return m, "after"
    else:
        m = find_ATen(data)
        if m != -1:
            return m, "after"
        else:
            m = find_torch(data)
            return m, "before"

def process_one_file(a_file : str):
    data = ''
    with open(a_file) as f:
        data = f.read()
    has_for_loop = for_loop_regex.findall(data)
    if not has_for_loop:
        return
    needs_header = has_for_loop and irange_header not in data

    if needs_header:
        pos, stype = find_header_insertion_point(data)
        # we do no change the file if do not know where to insert the head file
        # for now, since there are too many of them
        if pos == -1:
            return
        if stype == "after":
            data = data[0:pos] + "\n" + irange_header + data[pos:]
        else:
            data = data[0:pos] + irange_header + "\n" + data[pos:]

    start = 0
    new_data = ""
    for match in for_loop_regex.finditer(data):
        loop_text_begin, loop_text_end = match.span()
        loop_var = match.group(1)
        lower_bound = match.group(2)
        upper_bound = match.group(3)
        bracket = " {" if match.group(4) == "{" else ""
        if lower_bound == "0":
            replacement_loop = new_loop_zero.format(
                loop_var=loop_var, upper_bound=upper_bound, bracket=bracket
            )
        else:
            replacement_loop = new_loop_range.format(
                loop_var=loop_var,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                bracket=bracket,
            )
        old_loop = data[loop_text_begin : loop_text_end]
        new_data += data[start : loop_text_begin] + replacement_loop
        start = loop_text_end
    new_data += data[start:]

    with open(a_file, "w") as fout:
        fout.write(new_data)

#filetypes = ('.cpp', '.cc', '.h', '.hpp')
filetypes = ('.cpp', '.cc')
#target_path = '..'
target_path = '../aten'

excluded_files = ['../c10/util/ConstexprCrc.h',
    '../aten/src/ATen/core/jit_type.h',
    '../aten/src/ATen/native/Math.h',
    '../c10/util/variant.h',
    '../c10/util/flags_use_no_gflags.cpp',
    '../caffe2/operators/cc_bmm_bg_op.h',
    '../aten/src/ATen/core/tensor_type.cpp',
    '../aten/src/ATen/native/Linear.cpp',
    '../aten/src/ATen/native/ConvolutionTBC.cpp',
    '../caffe2/share/fb/mask_rcnn/bbox_concat_batch_splits_op.h',
    '../aten/src/ATen/native/BatchLinearAlgebra.cpp',
    '../aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp',
    '../aten/src/ATen/native/cuda/DistributionTemplates.h',
    '../c10/util/sparse_bitset.h',
    '../torch/csrc/distributed/c10d/TCPStore.cpp',
    '../caffe2/fb/operators/calibration_op.h',
    '../torch/csrc/jit/testing/file_check.cpp',
    '../torch/csrc/jit/passes/concat_opt.cpp',
    '../torch/csrc/jit/tensorexpr/operators/reduction.cpp',
    '../torch/fb/operators/select_keys.cpp',
    '../torch/fb/operators/calibration/bucketize_calibration.cpp',
    '../fb/custom_ops/maskrcnn/int8/int8_aabb_roi_align.cpp',
    '../fb/custom_ops/maskrcnn/aabb/aabb_roi_align.cpp',
    '../caffe2/fb/tests/RecordIOHelper.cpp',
    '../test/cpp/api/rnn.cpp',
    '../torch/fb/training_toolkit/common/tdigest/tests/TestBufferedTDigest.cpp'
    ]

for current_folder, subfolders, files in os.walk(target_path):
    for a_file in files:
        if a_file.endswith(filetypes) and current_folder != '../caffe2/torch/jit':
            full_path = os.path.join(current_folder, a_file)
            if full_path not in excluded_files:
                process_one_file(full_path)