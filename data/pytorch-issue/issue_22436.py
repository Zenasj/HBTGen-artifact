import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()

args = get_arguments()

def main():
    """Create the model and start the training."""
    torch.manual_seed(0)

    cudnn.enabled = True
    cudnn.deterministic = True

    # Create network    
    class LinearNet(nn.Module):
        def __init__(self, input_nc, output_nc):
            super(LinearNet, self).__init__()
            self.l1 = nn.Linear(
                    input_nc,
                    output_nc
                )
            self.l2 = nn.Linear(
                    output_nc,
                    output_nc
                )
        def forward(self, x):
            x = self.l1(x)
            y = self.l2(x)
            return x, y
            
    num_gpus = int(torch.cuda.device_count())
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

    model = LinearNet(3, 2)
    model_D = LinearNet(2, 1)
    model = model.to(torch.device("cuda"))
    model_D = model_D.to(torch.device("cuda"))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            broadcast_buffers=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[args.local_rank], output_device=args.local_rank,
            broadcast_buffers=True, process_group=pg2
        )
    
    L1_loss = torch.nn.L1Loss(reduction="mean")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                          lr=1e-2, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-2, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    model.train()
    model_D.train()
    
    real_label = 1
    fake_label = 0
    
    for epoch in range(5):
        src_input_data = torch.randn((1,3)).cuda()
        src_label = torch.randn((1,2)).cuda()
        tgt_input_data = torch.randn((1,3)).cuda()

        optimizer.zero_grad()
        optimizer_D.zero_grad()
        model.zero_grad()
        model_D.zero_grad()
        
        # train generator
        fea, pred = model(src_input_data)
        loss1 = L1_loss(pred, src_label)
        loss1.backward()
        
        tgt_fea, tgt_pred = model(tgt_input_data)
        _, tgt_D_pred = model_D(tgt_fea)
        loss_adv = bce_loss(tgt_D_pred, tgt_D_pred.new_full(tgt_D_pred.size(), real_label))
        loss_adv.backward()
        
        optimizer.step()

        # # train discriminator
        # ...

if __name__ == '__main__':
    main()