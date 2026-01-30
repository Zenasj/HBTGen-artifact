import torch
import torch.nn as nn
import random
import sys

DEVICE = 'cuda'
SEQ_LEN_RANGE = (0, 1024)
BATCH_SIZE = 1024
VOCAB_SIZE = 1000
DIM = 256

emb = nn.Embedding(VOCAB_SIZE, DIM, device=DEVICE)

i = 0
max_vram_usage = 0
historical_max_seq_len = 0
print(f'{SEQ_LEN_RANGE=}\t{BATCH_SIZE=}\t{VOCAB_SIZE=}\t{DIM=}')
while True:
    i += 1
    ### create random input with random length ###
    batch_max_seq_len = random.randint(SEQ_LEN_RANGE[0], SEQ_LEN_RANGE[1])
    seqs = []
    for _ in range(BATCH_SIZE):
        seq = [random.randrange(VOCAB_SIZE) for _ in range(batch_max_seq_len)]
        seqs.append(seq)
    input_ids = torch.tensor(seqs, dtype=torch.int32, device=DEVICE)
    ### forward ###
    x = emb(input_ids)
    ### check current vram usage and log if increased ###
    vram_usage = torch.cuda.max_memory_reserved() // 1024 // 1024
    torch.cuda.reset_peak_memory_stats()
    historical_max_seq_len = max(historical_max_seq_len, batch_max_seq_len)
    if max_vram_usage < vram_usage:
        max_vram_usage = vram_usage
        print('\r\033[K'f'{i=}\t{max_vram_usage=}MB\t{historical_max_seq_len=}')
    ### show progress ###
    print('\r\033[K'f'[current {i=}, {vram_usage=}MB]', end='', file=sys.stderr, flush=True)