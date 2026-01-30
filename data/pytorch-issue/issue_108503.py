import torch
import torch.nn as nn
import torch.nn.functional as F

token_sequence = torch.randn(1, 2, 600, 80)

sub_sequence_1 = token_sequence[:, :, :128, :].cuda().bfloat16()
sub_sequence_2 = token_sequence[:, :, 128:256, :].cuda().bfloat16()
sub_sequence_3 = token_sequence[:, :, 256:384, :].cuda().bfloat16()
sub_sequence_4 = token_sequence[:, :, 543:, :].cuda().bfloat16()

sub_seq_1_self_att = F.scaled_dot_product_attention(sub_sequence_1, sub_sequence_1, sub_sequence_1)
sub_seq_2_self_att = F.scaled_dot_product_attention(sub_sequence_2, sub_sequence_2, sub_sequence_2)
sub_seq_3_self_att = F.scaled_dot_product_attention(sub_sequence_3, sub_sequence_3, sub_sequence_3)
sub_seq_4_self_att = F.scaled_dot_product_attention(sub_sequence_4, sub_sequence_4, sub_sequence_4)

#create a mask to do all sequences in parallel
mask = torch.zeros(1, 2, 600, 600)
#change mask to be all False
mask = mask.bool()
#let sub_sequence_1 only attend to sub_sequence_1
mask[:, :, :128, :128] = True
#let sub_sequence_2 only attend to sub_sequence_2
mask[:, :, 128:256, 128:256] = True
#let sub_sequence_3 only attend to sub_sequence_3
mask[:, :, 256:384, 256:384] = True
#let sub_sequence_4 only attend to sub_sequence_4
mask[:, :, 543:, 543:] = True


seq_self_att2 = F.scaled_dot_product_attention(token_sequence.cuda().bfloat16(), token_sequence.cuda().bfloat16(), token_sequence.cuda().bfloat16(), attn_mask=mask.cuda().bfloat16())

#check difference
print((torch.abs(sub_seq_2_self_att - seq_self_att2[:, :, 128:256, :]).max()))

#check if they're close
print(torch.allclose(sub_seq_2_self_att, seq_self_att2[:, :, 128:256, :], atol=1e-6))
print(torch.allclose(sub_seq_1_self_att, seq_self_att2[:, :, :128, :], atol=1e-6))
print(torch.allclose(sub_seq_3_self_att, seq_self_att2[:, :, 256:384, :], atol=1e-6))
print(torch.allclose(sub_seq_4_self_att, seq_self_att2[:, :, 543:, :], atol=1e-6))

tensor(0.5547, device='cuda:0', dtype=torch.bfloat16)
False
False
False
False

Python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

def main():
    token_sequence = torch.randn(1, 1, 600, 80)

    sub_sequence_1 = token_sequence[:, :, :128, :].cuda().bfloat16()
    sub_sequence_2 = token_sequence[:, :, 128:256, :].cuda().bfloat16()
    sub_sequence_3 = token_sequence[:, :, 256:384, :].cuda().bfloat16()
    sub_sequence_4 = token_sequence[:, :, 543:, :].cuda().bfloat16()

    sub_seq_1_self_att = F.scaled_dot_product_attention(sub_sequence_1, sub_sequence_1, sub_sequence_1, scale=1)
    sub_seq_2_self_att = F.scaled_dot_product_attention(sub_sequence_2, sub_sequence_2, sub_sequence_2, scale=1)
    sub_seq_3_self_att = F.scaled_dot_product_attention(sub_sequence_3, sub_sequence_3, sub_sequence_3, scale=1)
    sub_seq_4_self_att = F.scaled_dot_product_attention(sub_sequence_4, sub_sequence_4, sub_sequence_4, scale=1)

    #create a mask to do all sequences in parallel
    mask = torch.zeros(1, 1, 600, 600)
    #change mask to be all False
    mask = mask.bool()
    #let sub_sequence_1 only attend to sub_sequence_1
    mask[:, :, :128, :128] = True
    #let sub_sequence_2 only attend to sub_sequence_2
    mask[:, :, 128:256, 128:256] = True
    #let sub_sequence_3 only attend to sub_sequence_3
    mask[:, :, 256:384, 256:384] = True
    #let sub_sequence_4 only attend to sub_sequence_4
    mask[:, :, 543:, 543:] = True


    seq_self_att2 = F.scaled_dot_product_attention(token_sequence.cuda().bfloat16(), token_sequence.cuda().bfloat16(), token_sequence.cuda().bfloat16(), attn_mask=mask.cuda().bfloat16(), scale=1)

    #check difference
    print((torch.abs(sub_seq_2_self_att - seq_self_att2[:, :, 128:256, :]).max()))

    #check if they're close
    torch.testing.assert_close(sub_seq_2_self_att, seq_self_att2[:, :, 128:256, :])
    torch.testing.assert_close(sub_seq_1_self_att, seq_self_att2[:, :, :128, :])
    torch.testing.assert_close(sub_seq_3_self_att, seq_self_att2[:, :, 256:384, :])
    torch.testing.assert_close(sub_seq_4_self_att, seq_self_att2[:, :, 543:, :],)

if __name__ == "__main__":
    main()