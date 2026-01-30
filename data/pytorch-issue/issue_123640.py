import random

import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity, schedule
from torch.profiler import ExecutionTraceObserver
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform GPU matrix multiplication and profiling."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to store the final result file."
    )
    return parser.parse_args()


def trace_handler(prof, output_dir):
    output_path = os.path.join(output_dir, "kineto_trace.json")
    prof.export_chrome_trace(output_path)


def gpu_matrix_multiplication(matrix1: np.ndarray, matrix2: np.ndarray) -> torch.Tensor:
    """
    Perform matrix multiplication on the GPU using PyTorch.

    Args:
        matrix1 (np.ndarray): The first input matrix as a NumPy array.
        matrix2 (np.ndarray): The second input matrix as a NumPy array.

    Returns:
        torch.Tensor: The result of the matrix multiplication, as a PyTorch tensor.

    Raises:
        ValueError: If matrices have incompatible shapes for multiplication.
    """
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Matrices have incompatible shapes for multiplication.")

    matrix1_torch = torch.tensor(matrix1, dtype=torch.float)
    matrix2_torch = torch.tensor(matrix2, dtype=torch.float)

    if torch.cuda.is_available():
        matrix1_torch = matrix1_torch.to('cuda')
        matrix2_torch = matrix2_torch.to('cuda')

    result_gpu = torch.matmul(matrix1_torch, matrix2_torch)

    return result_gpu


if __name__ == "__main__":
    args = parse_arguments()

    et = ExecutionTraceObserver()
    et_path = os.path.join(args.output_dir, "pytorch_et.json")
    et.register_callback(et_path)

    matrix_a = np.random.rand(1024, 1024)
    matrix_b = np.random.rand(1024, 1024)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=10, active=1),
        on_trace_ready=lambda prof: trace_handler(prof, args.output_dir)
    ) as prof:
        for epoch in range(20):
            result_on_gpu = gpu_matrix_multiplication(matrix_a, matrix_b)
            if epoch == 11:
                et.stop()
            if epoch == 10:
                et.start()
            prof.step()

    et.unregister_callback()

{
  "schema": "1.0.2-chakra.0.0.4", "pid": 70610, "time": "2024-04-09 08:56:19", "start_ts": 1128488051,
  "nodes": [
    {
      "id": 2, "name": "[pytorch|profiler|execution_trace|thread]", "ctrl_deps": 1,
      "inputs": {"values": [], "shapes": [], "types": []},
      "outputs": {"values": [], "shapes": [], "types": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 0}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 7}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": ""}]
    },
    {
      "id": 6, "name": "aten::lift_fresh", "ctrl_deps": 3,
      "inputs": {"values": [[4,5,0,1048576,8,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(double)"]},
      "outputs": {"values": [[4,5,0,1048576,8,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(double)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 2}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::lift_fresh(Tensor(a) self) -> Tensor(a)"}]
    },
    {
      "id": 9, "name": "aten::empty_strided", "ctrl_deps": 8,
      "inputs": {"values": [[1024,1024],[1024,1],6,0,"cpu",false], "shapes": [[[],[]],[[],[]],[],[],[],[]], "types": ["GenericList[Int,Int]","GenericList[Int,Int]","Int","Int","Device","Bool"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 5}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}]
    },
    {
      "id": 12, "name": "aten::copy_", "ctrl_deps": 8,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"],[4,5,0,1048576,8,"cpu"],false], "shapes": [[1024,1024],[1024,1024],[]], "types": ["Tensor(float)","Tensor(double)","Bool"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 6}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)"}]
    },
    {
      "id": 8, "name": "aten::_to_copy", "ctrl_deps": 7,
      "inputs": {"values": [[4,5,0,1048576,8,"cpu"],6,"<None>","cpu","<None>",false,"<None>"], "shapes": [[1024,1024],[],[],[],[],[],[]], "types": ["Tensor(double)","Int","None","Device","None","Bool","None"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 4}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"}]
    },
    {
      "id": 7, "name": "aten::to", "ctrl_deps": 3,
      "inputs": {"values": [[4,5,0,1048576,8,"cpu"],"cpu",6,false,true,"<None>"], "shapes": [[1024,1024],[],[],[],[],[]], "types": ["Tensor(double)","Device","Int","Bool","Bool","None"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 3}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"}]
    },
    {
      "id": 14, "name": "detach_", "ctrl_deps": 13,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [], "shapes": [], "types": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 8}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": ""}]
    },
    {
      "id": 13, "name": "aten::detach_", "ctrl_deps": 3,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 7}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::detach_(Tensor(a!) self) -> Tensor(a!)"}]
    },
    {
      "id": 16, "name": "aten::lift_fresh", "ctrl_deps": 3,
      "inputs": {"values": [[4,15,0,1048576,8,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(double)"]},
      "outputs": {"values": [[4,15,0,1048576,8,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(double)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 9}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::lift_fresh(Tensor(a) self) -> Tensor(a)"}]
    },
    {
      "id": 19, "name": "aten::empty_strided", "ctrl_deps": 18,
      "inputs": {"values": [[1024,1024],[1024,1],6,0,"cpu",false], "shapes": [[[],[]],[[],[]],[],[],[],[]], "types": ["GenericList[Int,Int]","GenericList[Int,Int]","Int","Int","Device","Bool"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 12}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}]
    },
    {
      "id": 22, "name": "aten::copy_", "ctrl_deps": 18,
      "inputs": {"values": [[20,21,0,1048576,4,"cpu"],[4,15,0,1048576,8,"cpu"],false], "shapes": [[1024,1024],[1024,1024],[]], "types": ["Tensor(float)","Tensor(double)","Bool"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 13}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)"}]
    },
    {
      "id": 18, "name": "aten::_to_copy", "ctrl_deps": 17,
      "inputs": {"values": [[4,15,0,1048576,8,"cpu"],6,"<None>","cpu","<None>",false,"<None>"], "shapes": [[1024,1024],[],[],[],[],[],[]], "types": ["Tensor(double)","Int","None","Device","None","Bool","None"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 11}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"}]
    },
    {
      "id": 17, "name": "aten::to", "ctrl_deps": 3,
      "inputs": {"values": [[4,15,0,1048576,8,"cpu"],"cpu",6,false,true,"<None>"], "shapes": [[1024,1024],[],[],[],[],[]], "types": ["Tensor(double)","Device","Int","Bool","Bool","None"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 10}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"}]
    },
    {
      "id": 24, "name": "detach_", "ctrl_deps": 23,
      "inputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [], "shapes": [], "types": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 15}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": ""}]
    },
    {
      "id": 23, "name": "aten::detach_", "ctrl_deps": 3,
      "inputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 14}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::detach_(Tensor(a!) self) -> Tensor(a!)"}]
    },
    {
      "id": 28, "name": "aten::resolve_conj", "ctrl_deps": 26,
      "inputs": {"values": [[4,27,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [[4,27,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 18}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::resolve_conj(Tensor(a) self) -> Tensor(a)"}]
    },
    {
      "id": 29, "name": "aten::resolve_conj", "ctrl_deps": 26,
      "inputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 19}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::resolve_conj(Tensor(a) self) -> Tensor(a)"}]
    },
    {
      "id": 30, "name": "aten::resolve_conj", "ctrl_deps": 26,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "outputs": {"values": [[10,11,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 20}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::resolve_conj(Tensor(a) self) -> Tensor(a)"}]
    },
    {
      "id": 26, "name": "aten::mm", "ctrl_deps": 25,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"],[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024],[1024,1024]], "types": ["Tensor(float)","Tensor(float)"]},
      "outputs": {"values": [[4,27,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 17}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::mm(Tensor self, Tensor mat2) -> Tensor"}]
    },
    {
      "id": 25, "name": "aten::matmul", "ctrl_deps": 3,
      "inputs": {"values": [[10,11,0,1048576,4,"cpu"],[20,21,0,1048576,4,"cpu"]], "shapes": [[1024,1024],[1024,1024]], "types": ["Tensor(float)","Tensor(float)"]},
      "outputs": {"values": [[4,27,0,1048576,4,"cpu"]], "shapes": [[1024,1024]], "types": ["Tensor(float)"]},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 16}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": 0}, {"name": "scope", "type": "uint64", "value": 0}, {"name": "tid", "type": "uint64", "value": 1}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": "aten::matmul(Tensor self, Tensor other) -> Tensor"}]
    },
    {
      "id": 1, "name": "[pytorch|profiler|execution_trace|process]", "ctrl_deps": 1,
      "inputs": {"values": [], "shapes": [], "types": []},
      "outputs": {"values": [], "shapes": [], "types": []},
      "attrs": [{"name": "rf_id", "type": "uint64", "value": 0}, {"name": "fw_parent", "type": "uint64", "value": 0}, {"name": "seq_id", "type": "int64", "value": -1}, {"name": "scope", "type": "uint64", "value": 7}, {"name": "tid", "type": "uint64", "value": 0}, {"name": "fw_tid", "type": "uint64", "value": 0}, {"name": "op_schema", "type": "string", "value": ""}]
    }
  ],
  "finish_ts": 1128488107
}

{
  "schemaVersion": 1,
  "traceEvents": [
  {
    "ph": "X", "cat": "user_annotation", "name": "ProfilerStep#10", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482786, "dur": 2674,
    "args": {
      "External id": 1,"Record function id": 0, "Ev Idx": 0
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::lift_fresh", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482806, "dur": 1,
    "args": {
      "External id": 2,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 1
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::to", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482818, "dur": 189,
    "args": {
      "External id": 3,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 2
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_to_copy", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482840, "dur": 167,
    "args": {
      "External id": 4,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 3
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::empty_strided", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482841, "dur": 1,
    "args": {
      "External id": 5,"Record function id": 0, "Ev Idx": 4
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::copy_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379482843, "dur": 164,
    "args": {
      "External id": 6,"Record function id": 0, "Ev Idx": 5
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::detach_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483010, "dur": 1,
    "args": {
      "External id": 7,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 6
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "detach_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483010, "dur": 1,
    "args": {
      "External id": 8,"Record function id": 0, "Ev Idx": 7
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::lift_fresh", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483013, "dur": 1,
    "args": {
      "External id": 9,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 8
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::to", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483014, "dur": 551,
    "args": {
      "External id": 10,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 9
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_to_copy", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483014, "dur": 551,
    "args": {
      "External id": 11,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 10
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::empty_strided", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483015, "dur": 0,
    "args": {
      "External id": 12,"Record function id": 0, "Ev Idx": 11
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::copy_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483015, "dur": 550,
    "args": {
      "External id": 13,"Record function id": 0, "Ev Idx": 12
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::detach_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483567, "dur": 0,
    "args": {
      "External id": 14,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 13
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "detach_", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483567, "dur": 0,
    "args": {
      "External id": 15,"Record function id": 0, "Ev Idx": 14
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::matmul", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483582, "dur": 1501,
    "args": {
      "External id": 16,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 15
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483582, "dur": 1500,
    "args": {
      "External id": 17,"Record function id": 0, "Sequence number": 0, "Fwd thread id": 0, "Ev Idx": 16
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::resolve_conj", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483585, "dur": 0,
    "args": {
      "External id": 18,"Record function id": 0, "Ev Idx": 17
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::resolve_conj", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483585, "dur": 0,
    "args": {
      "External id": 19,"Record function id": 0, "Ev Idx": 18
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::resolve_conj", "pid": 70610, "tid": 13366669,
    "ts": 1712667379483585, "dur": 0,
    "args": {
      "External id": 20,"Record function id": 0, "Ev Idx": 19
    }
  },
  {
    "name": "thread_name", "ph": "M", "ts": 1712667379478199, "pid": 70610, "tid": 13366669,
    "args": {
      "name": "thread 13366669 ()"
    }
  },
  {
    "name": "thread_sort_index", "ph": "M", "ts": 1712667379478199, "pid": 70610, "tid": 13366669,
    "args": {
      "sort_index": 13366669
    }
  },
  {
    "name": "thread_name", "ph": "M", "ts": 1712667379478199, "pid": 70610, "tid": 13366669,
    "args": {
      "name": "thread 13366669 ()"
    }
  },
  {
    "name": "thread_sort_index", "ph": "M", "ts": 1712667379478199, "pid": 70610, "tid": 13366669,
    "args": {
      "sort_index": 13366669
    }
  },
  {
    "ph": "X", "cat": "Trace", "ts": 1712667379478076, "dur": 7395,
    "pid": "Spans", "tid": "PyTorch Profiler",
    "name": "PyTorch Profiler (0)",
    "args": {
      "Op count": 0
    }
  },
  {
    "name": "process_sort_index", "ph": "M", "ts": 1712667379478076,
    "pid": "Spans", "tid": 0,
    "args": {
      "sort_index": 536870912
    }
  },
  {
    "name": "Iteration Start: PyTorch Profiler", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace PyTorch Profiler", "ts": 1712667379478076
  },
  {
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": 1712667379485735
  }
  ],
  "traceName": "./kineto_trace.json"
}

with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=10, active=1),
        on_trace_ready=lambda prof: trace_handler(prof, args.output_dir),
        execution_trace_observer=et
    ) as prof:
        for epoch in range(20):
            result_on_gpu = gpu_matrix_multiplication(matrix_a, matrix_b)
            #if epoch == 11:
            #    et.stop()
            #if epoch == 10:
            #    et.start()
            prof.step()

    #et.unregister_callback()