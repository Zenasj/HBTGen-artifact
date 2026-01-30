import torch
import argparse

def main(rank):
    print('this is rank', rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    main(args.local_rank)