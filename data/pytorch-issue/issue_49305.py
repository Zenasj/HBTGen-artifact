# oom.py
import argparse
from time import sleep
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=False, type=int)
    parser.add_argument("--mode", default=False, type=str)
    args, _ = parser.parse_known_args()

    print(f"{args.local_rank} is starting")
    sleep(3)

    if args.mode == "oom":
        # emulate OOM in 2nd card
        if args.local_rank == 1:
            raise RuntimeError("OOM")

    if args.mode == "clean-finish":
        sleep(1)
        print(f"{args.local_rank} is cleanly finishing")
        sys.exit(0)

    while (True):
        # emulate long running process
        print(f"{args.local_rank} is running")
        sleep(1)
        
if __name__ == "__main__":
    main()