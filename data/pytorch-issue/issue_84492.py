import torch

if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
            ws, rank = _get_distributed_settings()
            if num_workers > 0:
                self.worker_init_fn = functools.partial(
                    _sharding_worker_init_fn, self.worker_init_fn, ws, rank)
            else:
                torch.utils.data.graph_settings.apply_sharding(self.dataset, ws, rank)

def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    world_size = dist.get_world_size()
    rank_id = dist.get_rank()
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    total_workers = info.num_workers
    datapipe = info.dataset