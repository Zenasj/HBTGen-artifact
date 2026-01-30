import torch.nn as nn

import unittest
import torch


class QuantizationIssue(unittest.TestCase):
    def setUp(self):
        self.input_fp32 = torch.tensor([
            [5.384835, 12.083044, 27.318186],
            [12.4770565, 27.580862, 22.327364],
            [26.136148, 22.721376, 4.859485],
        ], dtype=torch.float32)
        self.input_scale = 0.13133743405342102
        self.input_zero_point = 1
        self.input_q = torch.quantize_per_tensor(
            self.input_fp32, self.input_scale, self.input_zero_point, dtype=torch.quint8,
        )
        self.input_expected_int_repr = torch.tensor([
            [42, 93, 209], [96, 211, 171], [200, 174, 38]
        ], dtype=torch.uint8)

        self.weight_fp32 = torch.tensor([
            [-0.3740014433860779, -0.2629697620868683, -0.17531317472457886],
            [-0.14317242801189423, 0.2074539214372635, 0.31264182925224304],
            [0.09350036084651947, 0.34478259086608887, 0.17531317472457886],
        ], dtype=torch.float32)
        self.weight_zero_point = 0
        self.weight_scale = 0.0029218862764537334
        self.weight_q = torch.quantize_per_tensor(
            self.weight_fp32, self.weight_scale, self.weight_zero_point, dtype=torch.qint8,
        )
        self.weight_expected_int_repr = torch.tensor([
            [-128, -90, -60], [-49, 71, 107], [32, 118, 60]
        ], dtype=torch.int8)
        self.output_scale = 0.10707724094390869
        self.output_zero_point = 0
        self.expected_output_int_repr = 113

    # This check verifies test validity
    def test_consistent_input(self):
        self.assertTrue((self.input_q.int_repr() == self.input_expected_int_repr).all())

    # This check verifies test validity
    def test_consistent_weight(self):
        self.assertTrue((self.weight_q.int_repr() == self.weight_expected_int_repr).all())

    # This check verifies test validity
    def test_qnnpack_conv(self):
        torch.backends.quantized.engine ='qnnpack'
        self._run_convolution()

    # THIS IS THE TEST THAT DEMONSTRATES THE ISSUE
    def test_fbgemm_conv(self):
        torch.backends.quantized.engine ='fbgemm'
        self._run_convolution()

    # This check verifies test validity
    def test_manual_calculation(self):
        flat_input = self.input_expected_int_repr.reshape([-1]).int()
        flat_weight = self.weight_expected_int_repr.reshape([-1]).int()
        qsum = ((flat_input - self.input_zero_point) * (flat_weight - self.weight_zero_point)).sum()
        unquantized_result = qsum * self.input_scale * self.weight_scale / self.output_scale
        rounded_result = torch.round(unquantized_result).item()
        self.assertEqual(rounded_result, self.expected_output_int_repr)

    def _run_convolution(self):
        conv_layer = torch.nn.intrinsic.quantized.ConvReLU2d(
            1, 1, (3, 3), bias=False
        )
        conv_layer.scale = self.output_scale
        conv_layer.zero_point = self.output_zero_point
        conv_layer.set_weight_bias(self.weight_q[None, None, :, :], None)
        output = conv_layer(self.input_q[None, None, :, :])
        self.assertEqual(output.int_repr().item(), self.expected_output_int_repr)


if __name__ == '__main__':
    unittest.main()