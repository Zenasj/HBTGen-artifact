import torch

def test_dtensor_tensor_is_not_autograd_leaf_but_local_is_noncontiguous(self):

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=3, world_size=2
        )
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = torch.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [_Partial()], run_check=False, shape=(10, 257, 160), stride=(41120, 160, 1))
        tmp_dt = x_dt.redistribute(mesh, (Shard(1),))

        from torch._subclasses import FakeTensorMode
        m = FakeTensorMode()
        tmp_dt_fake = m.from_tensor(tmp_dt)
        self.assertEqual(tmp_dt.shape, tmp_dt_fake.shape)
        self.assertEqual(tmp_dt.stride(), tmp_dt_fake.stride())
        self.assertEqual(tmp_dt._local_tensor.shape, tmp_dt_fake._local_tensor.shape)
        # This assert **fails**
        # tmp_dt._local_tensor is not contiguous, but tmp_dt_fake._local_tensor advertises as contiguous
        self.assertEqual(tmp_dt._local_tensor.stride(), tmp_dt_fake._local_tensor.stride())