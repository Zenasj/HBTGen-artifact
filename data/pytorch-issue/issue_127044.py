py
store = FileStore(self.store_path, self.world_size)
healthcheck = HealthcheckNCCL(
    store=store,
    rank=self.rank,
    world_size=self.world_size,
    local_world_size=8,
    abort_on_error=True,
    interval=timedelta(seconds=60),
    timeout=timedelta(milliseconds=10),
)