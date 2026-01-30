import torch
from multiprocessing import Pool
import os

def process(data):
    print(f'{os.sched_getaffinity(0)}')


if __name__ == "__main__":
    pool = Pool(76)
    pool.map(process, list(range(5000)))