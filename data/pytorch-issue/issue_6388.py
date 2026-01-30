import torch

en_init_state=(Variable(torch.zeros((layers*bi,batch,hidden))),Variable(torch.zeros(            (layers*bi,batch,hidden))))
enc_final, memory_bank = self.encoder(src,encoder_state=en_init_state)