import argparse

import torch
import torch.nn as nn

from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="GPU device id"
    )
    return parser


def timing_cuda(model, num_batches, inputs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        _ = model(inputs)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event)) / num_batches


def benchmark(num_batches: int, batch_size: int, max_seqlen: int, use_half: bool, device_id: int):
    
    layers_vanilla = []
    layers_bt = []
    for i in range(12):
        vanilla_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)  # as bert-base-uncased
        bt_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)

        vanilla_layer.norm2.eps = 2e-5  # disable fastpath

        assert vanilla_layer.norm1.eps != vanilla_layer.norm2.eps

        layers_vanilla.append(vanilla_layer)
        layers_bt.append(bt_layer)
    
    vanilla_model = nn.Sequential(*layers_vanilla)
    bt_model = nn.Sequential(*layers_bt)

    inputs = torch.rand(batch_size, max_seqlen, 768)

    if use_half is True:
        vanilla_model = vanilla_model.half()
        bt_model = bt_model.half()
        inputs = inputs.half()

    vanilla_model = vanilla_model.eval().to(f"cuda:{device_id}")
    bt_model = bt_model.eval().to(f"cuda:{device_id}")
    inputs = inputs.to(f"cuda:{device_id}")

    # Warmup
    _ = vanilla_model(inputs)
    torch.cuda.synchronize()
    _ = bt_model(inputs)
    torch.cuda.synchronize()

    vanilla_time = timing_cuda(vanilla_model, num_batches, inputs)
    bt_time = timing_cuda(bt_model, num_batches, inputs)

    return vanilla_time, bt_time


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [32, 64, 128]
    SEQ_LEN = [16, 32, 64, 128, 256]

    output_file = open("log_transformerencoderlayer.py", "w")
    output_file.write(
        "num_batches, batch_size, seq_len, use half, Vanilla time (ms), BT time (ms), Speedup\n"
    )
    for bs in tqdm(BATCH_SIZES, desc="batch size"):
        for seq_len in tqdm(SEQ_LEN, desc="sequence length"):
            max_seqlen = seq_len
            vanilla_time, bt_time = benchmark(
                args.num_batches,
                bs,
                max_seqlen,
                args.use_half,
                args.device_id
            )

            speedup = vanilla_time / bt_time

            output_file.write(
                "{},{},{},{},{},{},{}\n".format(
                    args.num_batches,
                    bs,
                    seq_len,
                    args.use_half,
                    f"{vanilla_time:.2f}",
                    f"{bt_time:.2f}",
                    f"{speedup:.3f}",
                )
            )
    output_file.close()