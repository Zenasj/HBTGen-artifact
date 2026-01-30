nvidia-smi

if running_loss is None:
    running_loss = loss.item()
else:
    running_loss = running_loss * .99 + loss.item() * .01

def _stage2(self, xs):
        proposal, fm = xs
        if proposal.dim()==2 and proposal.size(1) == 5:
            # train mode
            roi = roi_align(fm, proposal, output_size=[15, 15])
        elif proposal.dim()==3 and proposal.size(2) == 4:
            # eval mode
            roi = roi_align(fm, [proposal[0]], output_size=[15, 15])
        else:
            assert AssertionError(" The boxes tensor shape should be Tensor[K, 5] in train or Tensor[N, 4] in eval")
        x = self.big_kernel(roi)
        cls = self.cls_fm(x)
        rgr = self.rgr_fm(x)
        return cls, rgr