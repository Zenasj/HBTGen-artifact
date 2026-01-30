import torch
import timeit

with torch.no_grad():
    device = "cuda"
    dim0 = 30
    dim1 = 120000
    arr1 = torch.zeros(dim0, dim1, device = device) # Create a tensor with dim0 much less than dim1
    # Add a few nonzero elements to the tensor
    arr1[0, 3] = 1
    arr1[0, 10] = 1
    arr1[0, 12089] = 1
 
    arr1 = arr1.to_sparse() # Convert to sparse COO tensor
    arr2 = torch.randn(dim0, 2048, device = device) # Make a dummy array to test matmuls

    print("Sparse Transposed: ", timeit.timeit("torch.matmul(torch.transpose(arr1, 0, 1), arr2)", globals=globals(), number=100))

    # Try again without any transpositions to isolate the issue
    arr1_pretransposed = torch.zeros(dim1, dim0, device = device)
    arr1_pretransposed[3, 0] = 1
    arr1_pretransposed[10, 0] = 1
    arr1_pretransposed[12089, 0] = 1
    arr1_pretransposed = arr1_pretransposed.to_sparse()
    print("Sparse not-Transposed: ", timeit.timeit("torch.matmul(arr1_pretransposed, arr2)", globals=globals(), number=100))

    # Compare transpose speed with dense tensors
    arr1_dense = arr1.to_dense()
    print("Dense Transposed: ", timeit.timeit("torch.matmul(torch.transpose(arr1_dense, 0, 1), arr2)", globals=globals(), number=100))

    # Try without transpositions on a dense tensor for completeness
    arr1_dense_pretransposed = arr1_pretransposed.to_dense()
    print("Dense not-Transposed: ", timeit.timeit("torch.matmul(arr1_dense_pretransposed, arr2)", globals=globals(), number=100))
    # Calculate sparsity percentage
    print(f"Sparsity percentage:  {torch.count_nonzero(arr1.to_dense()).item() / arr1.numel() * 100}%")