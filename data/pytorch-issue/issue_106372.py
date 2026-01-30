from typing import Any
import torch
from torch import nn
from torch import _dynamo
from torch._functorch.aot_autograd import aot_module_simplified
import functorch
from functorch.compile import make_boxed_func


class CustomBackend(object):
    pytorch_graph_passes = []
    forward_graph_passes = []
    backward_graph_passes = []

    def graph_processing_pytorch(self, gm, example_inputs):
        # graph transform (graph optimization) for the captured graph in pytorch graph
        for graph_pass in CustomBackend.pytorch_graph_passes:
            graph_pass(gm, example_inputs)

    def graph_processing_aot_forward(self, gm, example_inputs):
        # graph transform (graph optimization) for the captured graph in aot autograd forward graph
        for graph_pass in CustomBackend.forward_graph_passes:
            graph_pass(gm, example_inputs)

    def graph_processing_aot_backward(self, gm, example_inputs):
        # graph transform (graph optimization) for the captured graph in aot autograd backward graph
        for graph_pass in CustomBackend.backward_graph_passes:
            graph_pass(gm, example_inputs)

    def forward_compiler(self, gm, example_inputs):
        self.graph_processing_aot_forward(gm, example_inputs)
        return make_boxed_func(gm.forward)

    def backward_compiler(self, gm, example_inputs):
        self.graph_processing_aot_backward(gm, example_inputs)
        return make_boxed_func(gm.forward)


    def __call__(self, gm, example_inputs):
        self.graph_processing_pytorch(gm, example_inputs)
        return aot_module_simplified(
            gm, example_inputs,
            fw_compiler=self.forward_compiler,
            bw_compiler=self.backward_compiler
        )