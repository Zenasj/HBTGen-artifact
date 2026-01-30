3
import time
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

SEQUENCE_LENGTH = 512


def generate_random_batch(tokenizer, batch_size, sequence_length=SEQUENCE_LENGTH):
    """Generate a batch of random sequences for testing."""
    return tokenizer(
        [" ".join(["I am a test string."] * sequence_length) for _ in range(batch_size)],
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors="pt",
    )


def benchmark_throughput(model, tokenizer, mixed_precision=False, batch_size=32, num_iterations=3):
    """Test and print the throughput of the model."""
    input_data = generate_random_batch(tokenizer, batch_size).to(device)
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            # PyTorch only supports bfloat16 mixed precision on CPUs.
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=mixed_precision):
                _ = model(**input_data)

    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=mixed_precision):
                _ = model(**input_data)
    end_time = time.time()

    elapsed_time = end_time - start_time
    sequences_per_second = (batch_size * num_iterations) / elapsed_time
    latency = 1000 / sequences_per_second * batch_size

    return sequences_per_second, latency


if __name__ == "__main__":
    device = torch.device("cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device).eval()

    model = torch.compile(model)

    throughput, latency = benchmark_throughput(model, tokenizer, mixed_precision=False)
    print(f"FP32 Throughput: {throughput:.2f} sequences/second, Latency: {latency:.2f} ms")

    throughput, latency = benchmark_throughput(model, tokenizer, mixed_precision=True)
    print(f"Mixed Precision Throughput: {throughput:.2f} sequences/second, Latency: {latency:.2f} ms")

3
import torch
import time


def benchmark_tflops(matrix_size=1024, dtype=torch.float32, num_of_runs=10):
    # Create random matrices
    a = torch.randn(matrix_size, matrix_size, dtype=dtype, device="cpu")
    b = torch.randn(matrix_size, matrix_size, dtype=dtype, device="cpu")

    with torch.no_grad():
        for _ in range(num_of_runs):
            # Warm-up: Run matrix multiplication once to initialize kernels and memory
            torch.mm(a, b)

    start_time = time.time()
    with torch.no_grad():
        # Perform matrix multiplication
        for _ in range(num_of_runs):
            c = torch.mm(a, b)
    time_taken = time.time() - start_time

    # Calculate TFlops
    tflops = num_of_runs * (2.0 * matrix_size**3) / (time_taken * 1e12)
    return tflops


if __name__ == "__main__":
    matrix_size = 1024
    num_of_runs = 10
    tflops = benchmark_tflops(matrix_size=matrix_size, dtype=torch.float32, num_of_runs=num_of_runs)
    print(f"Matrix multiplication for size {matrix_size}x{matrix_size}, FP32 achieved: {tflops:.4f} TFlops")

    tflops = benchmark_tflops(matrix_size=matrix_size, dtype=torch.float16, num_of_runs=num_of_runs)
    print(f"Matrix multiplication for size {matrix_size}x{matrix_size}, FP16 achieved: {tflops:.4f} TFlops")

    tflops = benchmark_tflops(matrix_size=matrix_size, dtype=torch.bfloat16, num_of_runs=num_of_runs)
    print(f"Matrix multiplication for size {matrix_size}x{matrix_size}, BF16 achieved: {tflops:.4f} TFlops")