from time import sleep
import torch

def main():
    print("Hello world")
    sleep(1000)

if __name__ == "__main__":
    main()

import os

import psutil


def main() -> None:
    print(psutil.Process(os.getpid()).memory_info())
    
    import torch
    
    print(torch.__version__)
    
    print(psutil.Process(os.getpid()).memory_info())


if __name__ == '__main__':
    main()