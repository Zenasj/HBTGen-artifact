dist.reduce_scatter_tensor(buffer, data, group=local_pg)
dist.all_reduce(buffer, group=cross_pg)
dist.all_gather_into_tensor(data, buffer, group=local_pg)

dist.reduce_scatter_tensor(buffer, data, group=local_pg, async_op=True)
dist.all_reduce(buffer, group=cross_pg, async_op=True)
dist.all_gather_into_tensor(data, buffer, group=local_pg, async_op=True)

nccl_options = dist.ProcessGroupNCCL.Options(stream=user_stream)
pg = dist.new_group(backend="nccl", pg_options=nccl_options)

pg1.set_stream(user_stream)
pg2.set_stream(user_stream)
dist.reduce_scatter_tensor(buffer, data, group=pg1, async_op=True)
dist.all_reduce(buffer, group=pg2, async_op=True)

dist.reduce_scatter_tensor(buffer, data, group=local_pg)
dist.all_reduce(buffer, group=cross_pg)
dist.all_gather_into_tensor(data, buffer, group=local_pg)