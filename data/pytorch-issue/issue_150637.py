import torch

def matrix_vector_operations(N_values):
    for N in N_values:
            A = torch.rand(N, N, dtype=torch.float16, device="cpu")
            X = torch.rand(N, dtype=torch.float16, device="cpu")

            print(" Allocated tensors for N = ", N)
            B = torch.matmul(A, X)
            print(" Completed matmul for N = ", N)

if __name__ == "__main__":
    N_values = [1, 10, 50000]  # Define different N values

    matrix_vector_operations(N_values)