import torch

recv_infos = tuple([info for info in recv_infos if isinstance(info, _RecvInfo)])

composite_args = self._retrieve_recv_activations() + args

if self._output_merge_spec is None:
    for chunk in self._stage.output_chunks:
        for tensor in chunk:
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 0:
                tensor.unsqueeze_(0)