import torch
import torch.nn as nn

class CFlow(torch.nn.Module):

    def __init__(self, c):
        super(CFlow, self).__init__()
        L = c.pool_layers
        self.encoder, self.pool_layers, self.pool_dims = load_encoder_arch(c, L)
        self.encoder = self.encoder.to(c.device).eval()
        self.decoders = [load_decoder_arch(c, pool_dim) for pool_dim in self.pool_dims]
        self.decoders = [decoder.to(c.device) for decoder in self.decoders]
        params = list(self.decoders[0].parameters())
        for l in range(1, L):
            params += list(self.decoders[l].parameters())
        # optimizer
        self.optimizer = torch.optim.Adam(params, lr=c.lr)
        self.N=256

    def forward(self, x):
        P = c.condition_vec
        #print(self.decoders)
        self.decoders = [decoder.eval() for decoder in self.decoders]
        height = list()
        width = list()
        i=0
        test_dist = [list() for layer in self.pool_layers]
        test_loss = 0.0
        test_count = 0
        start = time.time()
        _ = self.encoder(x)
        with torch.no_grad():
            for l, layer in enumerate(self.pool_layers):
                e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H * W
                E = B * S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC

                decoder = self.decoders[l]
                FIB = E // self.N + int(E % self.N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * self.N, (f + 1) * self.N)
                    else:
                        idx = torch.arange(f * self.N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    # m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
        return height, width, test_dist

model = CFlow(c)
print("Created !!!")
PATH = 'weights/mvtec_mobilenet_v3_large_freia-cflow_pl3_cb8_inp256_run0_Model_2022-11-08-10:50:39.pt'
model=load_weights(model,PATH)
model.eval()
batch_size = 1
x = torch.randn(batch_size, 3, 256, 256).to(c.device)
out=model(x)

def load_weights(model, filename):

    path = os.path.join(filename)
    state = torch.load(path)
    model.encoder.load_state_dict(state['encoder_state_dict'], strict=False)

    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in
                zip(model.decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))
    return model