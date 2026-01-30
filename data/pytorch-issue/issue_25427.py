import torch

self.assertEqual(
                2 * [torch.Tensor([(i * self.world_size) + (self.world_size * (self.world_size - 1) / 2)])],
                inputs[i],
                "Mismatch in interation {}".format(i)
            )

def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

self.assertLessEqual(max_err, prec, message)

self.assertEqual(
                2 * [torch.Tensor([(i * self.world_size) + (self.world_size * (self.world_size - 1) / 2)])],
                inputs[i],
                message="Mismatch in interation {}".format(i)
            )