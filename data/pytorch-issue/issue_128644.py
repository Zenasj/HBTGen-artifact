import torch
import numpy as np

def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        #Hp = int(np.ceil(H / self.window_size)) * self.window_size
        #Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # calculate attention mask for SW-MSA
        H_tensor = torch.tensor(H, dtype=torch.float32, device=x.device)
        W_tensor = torch.tensor(W, dtype=torch.float32, device=x.device)
        Hp = torch.ceil(H_tensor / self.window_size) * self.window_size
        Wp = torch.ceil(W_tensor / self.window_size) * self.window_size

        Hp = Hp.to(torch.int32)  # Ensure Hp is an integer tensor
        Wp = Wp.to(torch.int32)  # Ensure Wp is an integer tensor
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1