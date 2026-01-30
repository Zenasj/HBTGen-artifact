import torch

if args.local_rank == -1 or args.no_cuda:
        #device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #args.n_gpu = torch.cuda.device_count()
        device = torch.device("mps")
        args.n_gpu = 1
        print(device)