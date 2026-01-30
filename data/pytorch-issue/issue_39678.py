py
print("before chunk", per_shard_bag_offsets_stacked)

chunked = per_shard_bag_offsets_stacked.split(1) 
# OR
chunked = per_shard_bag_offsets_stacked.chunk(4) 

print("after chunk", chunked)