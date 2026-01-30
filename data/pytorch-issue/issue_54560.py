for i in range(world_size):
        if i != rank:
            options.set_device_map("worker" + str(rank), {rank: i})

for i in range(world_size):
        if i != rank:
            options.set_device_map("worker" + str(i), {rank: i})