import torch.nn as nn

import time
import torch


def main():
    # Embedding dim = 128, number of heads = 2.
    self_attn = torch.nn.modules.activation.MultiheadAttention(128, 2, dropout=0.1, batch_first=True)
    multihead_attn = torch.nn.modules.activation.MultiheadAttention(128, 2, dropout=0.1, batch_first=True)

    # Inference mode.
    self_attn.eval()
    multihead_attn.eval()

    # Determines the sequence length for the query tensor.
    seqlen = 1

    total_time = 0

    # Run six iterations of self-attn and multihead-attn.
    # This for loop emulates six runs of an auto-regressive 1-layer decoder.
    for i in range(1, 7):
        target = torch.rand([1, i, 128])  # B, sequence length, E
        target_last = torch.rand([1, seqlen, 128])  # B, seqlen, E. This is typically the last token in the target sequence.
        memory = torch.rand([1, 7, 128])  # B, seqlen, E

        start = time.time()
        self_attn(target_last, target, target, need_weights=False)
        multihead_attn(target_last, memory, memory, need_weights=False)
        total_time += time.time() - start

    print(f"time: {total_time}")


if __name__ == '__main__':
    main()