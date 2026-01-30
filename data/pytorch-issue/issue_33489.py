import torch.nn as nn

import numpy as np
import torch
import torchvision
import onnxruntime
import onnx

import io
import copy

import argparse


def ort_test_with_input(ort_sess, input, output, rtol, atol):
    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    inputs = list(map(to_numpy, input))
    outputs = list(map(to_numpy, output))

    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(model,
                   batch_size=2,
                   state_dict=None,
                   input=None,
                   use_gpu=True,
                   rtol=0.001,
                   atol=1e-7,
                   example_outputs=None,
                   do_constant_folding=True,
                   dynamic_axes=None,
                   test_with_inputs=None,
                   input_names=None,
                   output_names=None,
                   fixed_batch_size=False,
                   save_and_read_from_disk=False):
    model.eval()

    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input, )
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_copy = copy.deepcopy(input)
        output = model(*input_copy)
        if isinstance(output, torch.Tensor):
            output = (output, )

        # export the model to ONNX
        if save_and_read_from_disk:
            f = "mrcnn_test.onnx"
        else:
            f = io.BytesIO()
        input_copy = copy.deepcopy(input)
        torch.onnx.export(model,
                          input_copy,
                          f,
                          opset_version=11,
                          example_outputs=output,
                          do_constant_folding=True,
                          dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names)

        # compute onnxruntime output prediction
        if save_and_read_from_disk:
            print('checking model: {} file'.format(f))
            loaded_model = onnx.load(f)
            onnx.checker.check_model(loaded_model)
            ort_sess = onnxruntime.InferenceSession(loaded_model)
        else:
            ort_sess = onnxruntime.InferenceSession(f.getvalue())
        input_copy = copy.deepcopy(input)
        ort_test_with_input(ort_sess, input_copy, output, rtol, atol)

        # if additional test inputs are provided run the onnx
        # model with these inputs and check the outputs
        if test_with_inputs is not None:
            for test_input in test_with_inputs:
                if isinstance(test_input, torch.Tensor):
                    test_input = (test_input, )
                test_input_copy = copy.deepcopy(test_input)
                output = model(*test_input_copy)
                if isinstance(output, torch.Tensor):
                    output = (output, )
                ort_test_with_input(ort_sess, test_input, output, rtol, atol)


def run(save_and_read_from_disk=False, mrcnn=False):
    if mrcnn:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
    else:
        model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=200,
                                                                                 max_size=300)
    model.eval()
    x = torch.randn(2, 3, 200, 300, requires_grad=True)
    run_model_test(model,
                   batch_size=2,
                   input=(x, ),
                   use_gpu=True,
                   rtol=1e-3,
                   atol=1e-5,
                   do_constant_folding=True,
                   dynamic_axes=None,
                   test_with_inputs=None,
                   input_names=None,
                   output_names=None,
                   fixed_batch_size=None,
                   save_and_read_from_disk=save_and_read_from_disk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests with ONNX')
    parser.add_argument('--save_and_read_from_disk',
                        action='store_true',
                        help='writes onnx to disk and then loads it to run test')
    parser.add_argument('--mrcnn', action='store_true', help='use maskrcnn or otherwise use fastercnn')
    args = parser.parse_args()
    run(save_and_read_from_disk=args.save_and_read_from_disk, mrcnn=args.mrcnn)