import torch
import torch.nn as nn

if __name__=='__main__':
    args = get_args_parser().parse_args()

    init_distributed_mode(args)
    with _ddp_replicated_tensor(False):
        model = M()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        tokenizer = RobertaTokenizerFast.from_pretrained('ckpt/roberta-base')
        
        text = ['I dont know','I know']

        tokenized = tokenizer(text, padding="longest", return_tensors="pt").to('cuda')
        ic(tokenized._encodings)
        ic(id(tokenized))
        tokenized,_ = model(tokenized = tokenized)
        ic(tokenized._encodings)
        ic(id(tokenized))