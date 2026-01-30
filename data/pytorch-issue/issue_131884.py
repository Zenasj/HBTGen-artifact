import torch.nn as nn

from __future__ import annotations

import random
import time
from typing import Literal

import numpy as np
import torch


def pad_sequence_with_flips(
    sequences: list[torch.Tensor],
    batch_first: bool = False,
    padding_value: int | float | bool = 0.0,
    padding_side: Literal["left", "right"] | str = "left",
) -> torch.Tensor:
    if padding_side == 'right':
        padded_sequence = torch._C._nn.pad_sequence([t.flatten() for t in sequences], batch_first=batch_first, padding_value=padding_value)
    elif padding_side=='left':
        padded_sequence = torch._C._nn.pad_sequence([t.flatten().flip(0) for t in sequences], batch_first=batch_first, padding_value=padding_value)  # pyright: ignore[reportArgumentType]
        padded_sequence = padded_sequence.flip(int(batch_first))
    else:
        raise ValueError(f"padding_side should be either 'right' or 'left', but got {padding_side}")

    return padded_sequence


sequence_lengths: list[int] = []

flip_left_pad_times: list[float] = []
flip_left_pad_times_std: list[float] = []

left_pad_times: list[float] = []
left_pad_times_std: list[float] = []

RUNS_PER_LOOP: int = 100


for i in range(1, 7):
    sequence_length = i * int(1e6) // 6
    sequence_lengths.append(sequence_length)

    sequences = [torch.randint(0, 2, (random.randint(1, sequence_length),), dtype=torch.bool) for _ in range(64)]

    inner_left_pad_times: list[float] = []
    inner_right_pad_times: list[float] = []

    inner_flip_left_pad_times: list[float] = []
    inner_flip_right_pad_times: list[float] = []

    for _ in range(RUNS_PER_LOOP):

        start = time.perf_counter()
        torch._C._nn.pad_sequence(sequences, batch_first=True, padding_value=False, padding_side="left")
        end = time.perf_counter()
        inner_left_pad_times.append(end - start)

        start = time.perf_counter()
        pad_sequence_with_flips(sequences, batch_first=True, padding_value=False, padding_side="left")
        end = time.perf_counter()
        inner_flip_left_pad_times.append(end - start)

    left_pad_times.append(sum(inner_left_pad_times) / len(inner_left_pad_times))
    left_pad_times_std.append(np.std(inner_left_pad_times))

    flip_left_pad_times.append(sum(inner_flip_left_pad_times) / len(inner_flip_left_pad_times))
    flip_left_pad_times_std.append(np.std(inner_flip_left_pad_times))

    print(f"Sequence Length: {sequence_length}, Left Pad Time: {left_pad_times[-1]}, Left with Flips Pad Time: {flip_left_pad_times[-1]}")


import matplotlib.pyplot as plt

plt.plot(sequence_lengths, left_pad_times, label="new pad_sequence left")
plt.scatter(sequence_lengths, left_pad_times)
plt.errorbar(sequence_lengths, left_pad_times, yerr=left_pad_times_std, linestyle='None', marker='^')

plt.plot(sequence_lengths, flip_left_pad_times, label="old pad_sequence left (2 flips)")
plt.scatter(sequence_lengths, flip_left_pad_times)
plt.errorbar(sequence_lengths, flip_left_pad_times, yerr=flip_left_pad_times_std, linestyle='None', marker='^')

plt.xlabel("Sequence Length")
plt.ylabel("Time (s)")
plt.legend(loc="upper right")

# Sequence Length: 166666, Left Pad Time: 0.06147645162009212, Left with Flips Pad Time: 0.09842291727001794
# Sequence Length: 333333, Left Pad Time: 0.08933195920990329, Left with Flips Pad Time: 0.15597836187991562
# Sequence Length: 500000, Left Pad Time: 0.08863158334006585, Left with Flips Pad Time: 0.15224887342999863
# Sequence Length: 666666, Left Pad Time: 0.10524682551997103, Left with Flips Pad Time: 0.18177212480995877
# Sequence Length: 833333, Left Pad Time: 0.11801802741003485, Left with Flips Pad Time: 0.20821274195001024
# Sequence Length: 1000000, Left Pad Time: 0.131894061660023, Left with Flips Pad Time: 0.23223503091008751