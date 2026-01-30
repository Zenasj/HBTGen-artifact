for i in range(100):
    dist.all_to_all_single(tensor_out, tensor_in)
dist.destroy_process_group()