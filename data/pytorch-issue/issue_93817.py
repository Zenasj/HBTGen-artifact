def run_worker(rank, model, lr, train_loader, device, epoch, weight, q):
    out_weight = rpc.rpc_sync(f"Worker{rank}", train, args=(rank, model, lr, train_loader, device, epoch, weight))
    q.put([rank, out_weight])