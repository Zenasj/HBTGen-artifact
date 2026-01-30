import torch

def test_on_completion_hook_exception(self):
        pg = self._get_process_group()

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            raise RuntimeError("hook error")

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        pg.broadcast([tensor]).wait()

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)