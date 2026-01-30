import torch
import torch.nn as nn
class Embed(nn.Module):
    def __init__(self,aa_probs):
        super(Embed, self).__init__()
        self.aa_probs = aa_probs
        self.embedding = nn.Embedding(num_embeddings=1, #----> does NOT matter the number
                                      embedding_dim=aa_probs,
                                      padding_idx=None,
                                      max_norm=None,
                                      norm_type=2.0,
                                      scale_grad_by_freq=False,
                                      sparse=False,
                                      _weight=None)
        self.cuda()
    def forward(self,input):
        output = self.embedding(input.type(torch.cuda.IntTensor))
        print(output.shape) #works
        print(output) #crashes
        return output

input = torch.tensor([[[1,2,-3],[4,-5,6],[-7,8,9]],[[10,-11,12],[13,-14,15],[-16,-17,18]]]).type(torch.cuda.DoubleTensor).cuda()

a = Embed(21)

a.forward(input)