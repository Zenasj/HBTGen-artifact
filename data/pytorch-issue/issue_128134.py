import torch
import numpy as np

def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))

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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows

(1, (u0//7), 7, 39, 7, 1)
torch.Size([1, u0, 273, 1])

for layer in self.layers_list[s]:
            print(output.dtype)
            print(output.shape)
            output, memories = layer(output,
                                        long_term_memories[idx] if
                                            long_term_memories is not None else None,
                                        short_term_memories[idx] if
                                            short_term_memories is not None else None,
                                        curr_id_emb=curr_id_embs[s] if
                                            curr_id_embs is not None else None,
                                        self_pos=self_pos[s] if 
                                            self_pos is not None else None,
                                        size_2d=sizes_2d[s])
            # decoder norm

def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):
        # Self-attention
        _tgt = self.norm1(tgt)
        if not self.use_self_pos:
            self_pos = None
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        if self.global_dilation > 1:
            k = k[::self.global_dilation,:,:]
            v = v[::self.global_dilation,:,:]
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(curr_QV, self.d_model, dim=2)
        curr_Q = curr_K = curr_QV[0]
        curr_V = curr_QV[1]

        if self.d_att is not None:
            curr_Q = self.linear_Qd(curr_Q)
        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            if self.d_att is not None:
                global_K = self.linear_Kd(global_K)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)

            if self.global_dilation>1 and self.memory_dilation:
                nhw,bs,ck = global_K.shape
                cv = global_V.shape[-1]
                # n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                if self.conv_dilation:
                    unfold_K = global_K.permute(1,2,0).reshape(bs,ck,size_2d[0],size_2d[1])
                    unfold_V = global_V.permute(1,2,0).reshape(bs,cv,size_2d[0],size_2d[1])
                    global_K = self.dilation_conv_K(unfold_K).reshape(bs,ck,-1).permute(2,0,1)
                    global_V = self.dilation_conv_V(unfold_V).reshape(bs,cv,-1).permute(2,0,1)
                else:
                    unfold_K = global_K.view(size_2d[0],size_2d[1],bs,ck)
                    unfold_V = global_V.view(size_2d[0],size_2d[1],bs,cv)
                    global_K = unfold_K[::d,::d,:,:].reshape(-1,bs,ck)
                    global_V = unfold_V[::d,::d,:,:].reshape(-1,bs,cv)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
        
        
        if self.memory_dilation:
            tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        else:
            if self.global_dilation>1:
                nhw,bs,ck = global_K.shape
                cv = global_V.shape[-1]
                n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                if self.conv_dilation:
                    unfold_K = global_K.permute(1,2,0).reshape(bs*n,ck,size_2d[0],size_2d[1])
                    unfold_V = global_V.permute(1,2,0).reshape(bs*n,cv,size_2d[0],size_2d[1])
                    dilated_K = self.dilation_conv_K(unfold_K).reshape(bs,ck,-1).permute(2,0,1)
                    dilated_V = self.dilation_conv_V(unfold_V).reshape(bs,cv,-1).permute(2,0,1)
                else:
                    unfold_K = global_K.view(n,size_2d[0],size_2d[1],bs,ck)
                    unfold_V = global_V.view(n,size_2d[0],size_2d[1],bs,cv)
                    dilated_K = unfold_K[:,::d,::d,:,:].reshape(-1,bs,ck)
                    dilated_V = unfold_V[:,::d,::d,:,:].reshape(-1,bs,cv)
            else:
                dilated_K,dilated_V = global_K,global_V
            tgt2 = self.long_term_attn(curr_Q, dilated_K, dilated_V)[0]

        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]