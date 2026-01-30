import torch

with mock.patch(
            "torch.distributed.fsdp._runtime_utils._post_backward_reshard",
            _post_backward_reshard_with_count,
        ):
            for step_num in range(2):
                if step_num == 1:
                    post_backward_reshard_count = 0
                    # changes requires grad to simulate post-backward hook re-registration behavior after n steps
                    seq[4]._flat_param.requires_grad_(True)
                inp = torch.randn((8, 5), device="cuda", requires_grad=inp_requires_grad)
                output = seq(inp)
                if step_num == 1:
                    seq.zero_grad()
                # note here that with the proposed update to ` _register_post_backward_hooks`, the previous
                # `requires_grad=False` path multi-grad post backward hook will have been removed and replaced with
                # appropriate `AccumulateGrad` object, otherwise, a `SystemError` will be thrown here on `step_num == 1`
                output.sum().backward()
                # If the input does not require gradient, then the 0th frozen
                # linear gets resharded in the catch-all reshard since we cannot
                # register an autograd hook on it
                expected_post_backward_reshard_count = (
                    self.NUM_LINEARS if inp_requires_grad else self.NUM_LINEARS - 1
                )
                self.assertEqual(
                    post_backward_reshard_count, expected_post_backward_reshard_count
                )