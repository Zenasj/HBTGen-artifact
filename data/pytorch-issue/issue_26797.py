import torch.nn as nn

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 16
seq_len = 50
target_len = seq_len//2
latent_dim = 128
vocab_size = 50

x = torch.randn((seq_len,batch_size,latent_dim)).cuda()
y = torch.randint(1,vocab_size,(batch_size*target_len,),dtype=torch.long).cuda()
x_len = seq_len*torch.ones((batch_size,),dtype=torch.long).cuda()
y_len = target_len*torch.ones((batch_size,),dtype=torch.long).cuda()
w = torch.nn.Linear(latent_dim,vocab_size).cuda()

def compute_ctc(x,y,x_len,y_len,use_cudnn):
    for p in w.parameters():
        if p.grad is not None:
            p.grad.zero_()
    # Forward
    output = w(x).log_softmax(dim=-1)
    if use_cudnn:
        loss = torch.nn.functional.ctc_loss(output,
                                            y.to('cpu',torch.int32),
                                            x_len.to('cpu',torch.int32),
                                            y_len.to('cpu',torch.int32))
    else:
        loss = torch.nn.functional.ctc_loss(output,y,x_len,y_len)
    # backward
    loss.backward()
    m, b = w.parameters()
    print('loss = {}\ngrad_norm = {}'.format(loss, m.grad.view(-1).norm()))
    return m.grad.clone()

print("===== Pytorch CTC =====")
torch_grad = compute_ctc(x,y,x_len,y_len,False)
print("===== Cudnn CTC =====")
cudnn_grad = compute_ctc(x,y,x_len,y_len,True)
print("===== Grad diff. =====")
print("Cos Sim. = ",torch.nn.functional.cosine_similarity(torch_grad.view(-1),cudnn_grad.view(-1),dim=0))
print("Magnitude  = ",cudnn_grad.view(-1).norm() / torch_grad.view(-1).norm())