dynamo_model = torch.compile(model_copy, backend="inductor")
    #dynamo_model = dynamo.optimize("inductor")(model_copy)

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

import torch._dynamo as dynamo

import argparse
import copy

from tqdm import tqdm
from transformers import AutoModel

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
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
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
    )
    return parser

def timing_cuda(model, num_batches, input_ids):
    print("model device:", model.device)
    print("input device:", input_ids.device)
    print(type(model))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for i in range(num_batches):
            _ = model(input_ids)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(model_name, num_batches, batch_size, sequence_length, is_cuda, is_half, use_mask):
    print("Loading model {}".format(model_name))
    hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if is_half else None).eval()

    if is_cuda:
        hf_model = hf_model.to(0)

    model_copy = copy.deepcopy(hf_model)

    dynamo_model = torch.compile(model_copy, backend="inductor")
    #dynamo_model = dynamo.optimize("inductor")(model_copy)

    hf_model.eval()
    dynamo_model.eval()

    vocab_size = 30522  #TODO: generalize
    input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long)

    if is_cuda:
        input_ids = input_ids.to(0)

    if use_mask:
        raise NotImplementedError()

    # Warmup
    _ = hf_model(input_ids)
    torch.cuda.synchronize()

    _ = dynamo_model(input_ids)
    torch.cuda.synchronize()

    print("input_ids:", input_ids)
    print("input_ids shape:", input_ids.shape)

    total_hf_time = timing_cuda(hf_model, num_batches, input_ids)
    total_dynamo_time = timing_cuda(dynamo_model, num_batches, input_ids)

    return total_dynamo_time, total_hf_time


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [8, 16]
    SEQ_LEN = [64, 128]
    #BATCH_SIZES = [8]
    #SEQ_LEN = [128]

    output_file = open("log_{}_compile.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write(
        "num_batches, batch_size, seq_len, is cuda, is half, HF time, dynamo time, Speedup\n"
    )
    for batch_size in tqdm(BATCH_SIZES, desc="Batch size"):
        for sequence_length in tqdm(SEQ_LEN, desc="Sequence length"):
            total_dynamo_time, total_hf_time = benchmark(
                args.model_name,
                args.num_batches,
                batch_size,
                sequence_length,
                args.use_cuda,
                args.use_half,
                args.use_mask,
            )

            speedup = total_hf_time / total_dynamo_time

            output_file.write(
                "{},{},{},{},{},{},{},{}\n".format(
                    args.num_batches,
                    batch_size,
                    sequence_length,
                    args.use_cuda,
                    args.use_half,
                    total_hf_time,
                    total_dynamo_time,
                    speedup,
                )
            )
    output_file.close()