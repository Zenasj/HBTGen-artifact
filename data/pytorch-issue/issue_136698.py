total_length = dump_chrome_trace(
        f,
        input,
        chrome_trace_file_name,
        optimize_ctx,
        [ProfilerActivity.CUDA],
        num_runs=num_runs,
        devices=["cuda"],  # Changed from "cuda" to ["cuda"]
    )