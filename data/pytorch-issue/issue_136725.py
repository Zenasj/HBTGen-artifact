import torch.distributed as dist
# Run on process 1 (server)
server_store = dist.TCPStore("127.0.0.1", 1234, world_size=0, is_master=True)
# Run on process 2 (client)
client_store = dist.TCPStore("127.0.0.1", 1234, world_size=1, is_master=True)
# Attempt to use the store methods from either the client or server after initialization
# This will cause a crash because the number of users (world_size) is either not enough or incorrectly specified.
server_store.set("first_key", "first_value")
client_store.get("first_key")