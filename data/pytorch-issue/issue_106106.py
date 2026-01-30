import gc

import torch

from torch.testing._internal.common_utils import TestCase
# from unittest import TestCase

class TestA(TestCase):
    def test_a_caching_pinned_memory_multi_gpu(self):
        cycles_per_ms = 1086075.2822836833

        t = torch.FloatTensor([1]).pin_memory(); print('first t', id(t))
        ptr = t.data_ptr()
        gpu_tensor0 = torch.cuda.FloatTensor([0], device=0); print('gpu_tensor0', id(gpu_tensor0))
        gpu_tensor1 = torch.cuda.FloatTensor([0], device=1); print('gpu_tensor1', id(gpu_tensor1))

        with torch.cuda.device(1):
            torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
            gpu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory(); print('second t', id(t))

        self.assertNotEqual(t.data_ptr(), ptr) ### This line creates massive objects that are not gc'ed fast enough.

        with torch.cuda.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(gpu_tensor1[0], 1)
        self.assertEqual(gpu_tensor0[0], 2)

    def test_b_cuda_device_memory_allocated(self):
        for o in gc.get_objects():
            if isinstance(o, torch.Tensor):
                print('existing tensor objects', o, id(o))
                # for frame in gc.get_referrers(o):
                #     print(str(frame)[:1024])
    
        from torch.cuda import memory_allocated
        device_count = torch.cuda.device_count()

        old_alloc = [memory_allocated(idx) for idx in range(device_count)]
        print('old_alloc', old_alloc)

        x = torch.ones(10, device="cuda:0"); print('x', id(x))

        new_alloc = [memory_allocated(idx) for idx in range(device_count)]
        print('new_alloc', new_alloc)

        try:
            self.assertGreater(new_alloc[0], old_alloc[0])
            self.assertEqual(new_alloc[1:], old_alloc[1:])
        except:
            for o in gc.get_objects():
                if isinstance(o, torch.Tensor):
                    print('existing tensor objects', o, id(o))
            raise

if __name__ == '__main__':
    import unittest
    unittest.main()

def test_b_cuda_device_memory_allocated(self):
        for o in gc.get_objects():
            if isinstance(o, torch.Tensor):
                print('existing tensor objects', o, id(o))
                # for frame in gc.get_referrers(o):
                #     print(str(frame)[:1024])