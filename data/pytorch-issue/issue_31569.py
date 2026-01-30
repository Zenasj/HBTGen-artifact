import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as ptm


def turn_off_relu_inplace(layer):
    for m in layer.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return layer


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = ptm.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.relu.inplace = False
        self.maxpool = resnet.maxpool

        self.res2 = turn_off_relu_inplace(resnet.layer1)
        self.res3 = turn_off_relu_inplace(resnet.layer2)
        self.res4 = turn_off_relu_inplace(resnet.layer3)

    def forward(self, x, mask_obj, mask_otr):
        mask_obj = torch.unsqueeze(mask_obj, dim=1).float()
        mask_otr = torch.unsqueeze(mask_otr, dim=1).float()
        mx = self.conv1(x) + self.conv1_m(mask_obj) + self.conv1_o(mask_otr)
        mx = self.bn1(mx)
        mc1 = self.relu(mx)   # 1/2, 64
        mr1 = self.maxpool(mc1)  # 1/4, 64
        mr2 = self.res2(mr1)   # 1/4, 256
        mr3 = self.res3(mr2) # 1/8, 512
        # with torch.no_grad():
        mr4 = self.res4(mr3) # 1/16, 1024
        return mr4, mr3, mr2, mc1


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = ptm.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.relu.inplace = False
        self.maxpool = resnet.maxpool

        self.res2 = turn_off_relu_inplace(resnet.layer1)
        self.res3 = turn_off_relu_inplace(resnet.layer2)
        self.res4 = turn_off_relu_inplace(resnet.layer3)

    def forward(self, x):
        qx = self.conv1(x)
        qx = self.bn1(qx)
        qc1 = self.relu(qx)  # 1/2, 64
        qr1 = self.maxpool(qc1)  # 1/4, 64
        qr2 = self.res2(qr1)  # 1/4, 256
        qr3 = self.res3(qr2)  # 1/8, 512
        qr4 = self.res4(qr3)  # 1/8, 1024
        return qr4, qr3, qr2, qc1


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, dim_in, dim_out, scale_factor=2):
        super(Refine, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, stride=1)
        self.Resblock1 = ResBlock(dim_out, dim_out)
        self.Resblock2 = ResBlock(dim_out, dim_out)
        self.scale_factor = scale_factor

    def forward(self, x, pre_fea):
        nx = self.Resblock1(self.conv(x))
        m = nx + F.interpolate(pre_fea, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.Resblock2(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, memory_in, memory_out, query_in, query_out):
        B, D_e, T, H, W = memory_in.size()
        _, D_o, _, _, _ = memory_out.size()

        memory_in = memory_in.view(B, D_e, T*H*W)
        memory_in = memory_in.transpose(1, 2)

        query_in = query_in.view(B, D_e, H*W)

        prob = torch.bmm(memory_in, query_in)
        prob = prob / math.sqrt(D_e)
        prob = F.softmax(prob, dim=1)

        mem_out = memory_out.view(B, D_o, T*H*W)
        memory = torch.bmm(mem_out, prob)
        memory = memory.view(B, D_o, H, W)

        out = torch.cat([memory, query_out], dim=1)

        return out, prob


class KeyValue(nn.Module):
    def __init__(self, dim_in, dim_key, dim_value):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(dim_in, dim_key, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(dim_in, dim_value, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        key = self.Key(x)
        value = self.Value(x)
        return key, value


class STMNet(nn.Module):
    def __init__(self):
        super(STMNet, self).__init__()
        self.Encoder_Memory = Encoder_M()
        self.Encoder_Query = Encoder_Q()

        self.KeyValue_Memory_r4 = KeyValue(dim_in=1024, dim_key=128, dim_value=512)
        self.KeyValue_Query_r4 = KeyValue(dim_in=1024, dim_key=128, dim_value=512)

        self.Memory = Memory()
        self.Decoder = Decoder(mdim=256)

    def Pad_memory(self, memorys, num_objects, K):
        pad_memorys = []
        for m in memorys:
            pad_memory = torch.zeros(1, K, m.size()[1], 1, m.size()[2], m.size()[3]).cuda()
            pad_memory[0, 1:num_objects+1, :, 0] = m
            pad_memorys.append(pad_memory)

        return pad_memorys

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = torch.zeros(1, K, H, W).cuda()
        em[0, 0] = torch.prod(1-ps, dim=0) # bg prob
        em[0, 1:num_objects+1] = ps
        em = torch.clamp(em, 1e-7, 1-1e-7)
        log = torch.log((em / (1-em)))
        return log


    def forward(self, xs, pre_mask, keys, values, frame_index, num_objects):
        """
        :param xs: [previous_frame, current_frame]  (1, 3, h, w) (1,3,h,w)
        :param pre_mask: previous_mask  (1, 10, h,w)
        :param keys: Keys  (1, 10, 128,T,h,w )
        :param values: Values  (1, 10, 512,T,h,w )
        :param frame_indexs: [1,2,...]  ()
        :param num_objects: num objects torch.tensor([2])
        :return: current_seg_mask, updated_keys, updated_values
        """
        num_objects = num_objects[0].item()
        pre_frame, cur_frame = xs # B, C, H ,W

        # memorize
        _, K, H, W = pre_mask.size()
        (pre_frame, pre_mask), pad = pad_divide_by([pre_frame, pre_mask], 16, (pre_frame.size()[2], pre_frame.size()[3]))
        B_list = {'frame':[], 'mask':[], 'other':[]}
        for o in range(1, num_objects+1):
            B_list['frame'].append(pre_frame)
            B_list['mask'].append(pre_mask[:, o])
            B_list['other'].append(
                (torch.sum(pre_mask[:, 1:o], dim=1) +
                 torch.sum(pre_mask[:, o + 1:num_objects + 1], dim=1)).clamp(0, 1)
            )
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)
        mres4, _, _, _ = self.Encoder_Memory(B_['frame'], B_['mask'], B_['other'])
        mk4, mv4 = self.KeyValue_Memory_r4(mres4)
        prev_key, prev_value = self.Pad_memory([mk4, mv4], num_objects=num_objects, K=K)

        # Concat previous Keys and Values
        if frame_index == 1:
            cur_key, cur_value = prev_key, prev_value
        else:
            cur_key, cur_value = torch.cat([keys, prev_key], dim=3), torch.cat([values, prev_value], dim=3)

        # segmentation
        _, K, dim_key, T, H, W = cur_key.size()
        [cur_frame], pad = pad_divide_by([cur_frame], 16, (cur_frame.size()[2], cur_frame.size()[3]))

        seg_r4, seg_r3, seg_r2, _ = self.Encoder_Query(cur_frame)
        qk4, qv4 = self.KeyValue_Query_r4(seg_r4)

        # expand to --- num_objects, c, h, w
        key_embedding, value_embedding = qk4.expand(num_objects, -1, -1, -1), qv4.expand(num_objects, -1, -1, -1)
        r3_embedding, r2_embedding = seg_r3.expand(num_objects, -1, -1, -1), seg_r2.expand(num_objects, -1, -1, -1)

        # memory select
        memory4, prob = self.Memory(cur_key[0, 1:num_objects+1], cur_value[0, 1:num_objects+1],
                                    key_embedding, value_embedding)
        logits = self.Decoder(memory4, r3_embedding, r2_embedding)

        ps = F.softmax(logits, dim=1)[:, 1]

        # 1, K, H, W
        logit = self.Soft_aggregation(ps, K)

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :].clone()
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]].clone()

        return logit, cur_key, cur_value


def pad_divide_by(in_list, divide_factor, in_size):
    out_list = []
    h, w = in_size
    if h % divide_factor > 0:
        new_h = h+divide_factor - h%divide_factor
    else:
        new_h = h

    if w % divide_factor > 0:
        new_w = w + divide_factor - w%divide_factor
    else:
        new_w = w

    lh, uh = int((new_h - h) /2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)

    pad_array = (int(lw), int(uw), int(lh), int(uh))

    for in_fea in in_list:
        out_list.append(F.pad(in_fea, pad_array))
    return out_list, pad_array