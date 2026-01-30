import torch

from torch.testing._internal.common_device_type import instantiate_device_type_tests
class ProcessGroupNCCLTest(DistributedTestBase):
 
    def test_empty_tensors(self,device):
        #store = c10d.FileStore(self.file_name, self.world_size)
        pg = self.create_pg(device)
        local_device_idx = self.rank_to_device(device)[self.rank][0]
 
        xs = [torch.FloatTensor([]).to(device)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())
 
        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())
 
        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())
 
        ys = [[torch.FloatTensor([]).to(device) for _ in range(self.world_size)]]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())
 
        ys = [torch.FloatTensor([]).to(device)]
        xs = [[torch.FloatTensor([]).to(device) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())
 
 
devices = ["cuda"]
if TEST_HPU:
    devices.append("hpu")
instantiate_device_type_tests(ProcessGroupNCCLTest,globals(),only_for=devices)