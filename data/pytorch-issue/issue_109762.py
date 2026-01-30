# redistribute to single world size
one_ws = DeviceMesh(dist.get_rank())
redist = dtensor.redistribute(one_ws)
whole_tensor = redist.to_local()

# we want to materialize shards on device_mesh:
unshard = shards.redistribute(device_mesh, [Replicate()])
whole_tensor = unshard.to_local()