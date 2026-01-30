import torch

if __name__ == '__main__':
    for device in ['cpu',]:
        for jit in [True,]:
            print("Testing device {}, JIT {}".format(device, jit))
            m = Model(device=device, jit=jit)
            m.trainer.model.eval() 
            m.trainer.model.bert = torch.jit.freeze(m.trainer.model.bert) 
            m.eval() # run model