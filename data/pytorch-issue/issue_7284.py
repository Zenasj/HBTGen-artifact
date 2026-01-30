import torch

def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result