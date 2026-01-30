import torch

def _wait_impl(self) -> List[List[int]]:
        # Can not use is_torchdynamo_compiling(), as every such condition should be independent for compilation with graph breaks.
        if isinstance(self._splits_awaitable, dist.Work):
            self._splits_awaitable.wait()

        ret = self._output_tensor.view(self.num_workers, -1).T.tolist()  # <------ u22 introduced here

        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            for i in range(len(ret)):
                for j in range(len(ret[i])):
                    torch._check_is_size(ret[i][j])   # <----------  my question: why the _check_is_size isn't enough??
                    torch._check(ret[i][j] > 0)   # <------ added by diff V1