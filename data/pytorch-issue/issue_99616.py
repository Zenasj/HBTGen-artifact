import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def forward(self, x, forward_pass='default', sharpen=True):
        if forward_pass == 'dm':
            features = self.backbone(x)
            out = self.dm_head(features)
            return out

        elif forward_pass == 'pcl':
            features = self.backbone(x)
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            prototypes = self.prototypes.clone().detach()
            if sharpen:
                logits_proto = torch.mm(q, prototypes.t()) / self.temprature
            else:
                logits_proto = torch.mm(q, prototypes.t())
            return q, logits_proto

        elif forward_pass == 'all':
            features = self.backbone(x)
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            prototypes = self.prototypes.clone().detach()
            if sharpen:
                logits_proto = torch.mm(q, prototypes.t()) / self.temprature
            else:
                logits_proto = torch.mm(q, prototypes.t())
            out_dm = self.dm_head(features)
            return out_dm, logits_proto, q

# net and net2 are two same networks, train net, and fix net2
net.train()
net2.eval()

with torch.no_grad():
    size_x1, size_x2, size_u1, size_u2 = inputs_x1.size(0), inputs_x2.size(0), inputs_u1.size(0), inputs_u2.size(0)
    inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)

    # using 'all':
    dm_outputs_1, proto_outputs_1, features_1 = net(inputs, forward_pass='all')
    dm_outputs_2, proto_outputs_2, features_2 = net2(inputs, forward_pass='all')

    # or using 'dm' + 'pcl' as following to replace the above two lines:
    # dm_outputs_1 = net(inputs, forward_pass='dm')
    # dm_outputs_2 = net2(inputs, forward_pass='dm')
    # features_1, proto_outputs_1 = net(inputs, forward_pass='pcl')
    # features_2, proto_outputs_2 = net2(inputs, forward_pass='pcl')

    dm_o_x11, dm_o_x12, dm_o_u11, dm_o_u12 = torch.split(dm_outputs_1, [size_x1, size_x2, size_u1, size_u2], dim=0)
    dm_o_x21, dm_o_x22, dm_o_u21, dm_o_u22 = torch.split(dm_outputs_2, [size_x1, size_x2, size_u1, size_u2], dim=0)

labels_x_soft = torch.zeros(batch_size, args.num_classes, device=device).scatter_(1, labels_x.view(-1, 1), 1)
w_x = w_x.view(-1, 1).type(torch.FloatTensor).to(device)

targets_u = co_guessing(dm_o_u11, dm_o_u12, dm_o_u21, dm_o_u22, args.T)
targets_x = co_refinement(dm_o_x11, dm_o_x12, labels_x_soft, w_x, args.T)

variable_dict = {'inputs_x1': inputs_x1, 'inputs_x2': inputs_x2, 'inputs_u1': inputs_u1,
                 'inputs_u2': inputs_u2, 'targets_x': targets_x, 'targets_u': targets_u}
loss_dm = dividemix_train_step(args, net, 'dm', variable_dict, dm_criterion, batch_size, batch_idx, num_iter, epoch, device)

def co_guessing(outputs_u11, outputs_u12, outputs_u21, outputs_u22, T):
    with torch.no_grad():
        # label co-guessing of unlabeled samples
        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
              torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
        ptu = pu ** (1 / T)  # temperature sharpening

        targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
        targets_u = targets_u.detach()
    return targets_u


def co_refinement(outputs_x11, outputs_x12, labels_x_soft, w_x, T):
    with torch.no_grad():
        # label refinement of labeled samples
        px = (torch.softmax(outputs_x11, dim=1) + torch.softmax(outputs_x12, dim=1)) / 2
        px = w_x * labels_x_soft + (1 - w_x) * px
        ptx = px ** (1 / T)  # temperature sharpening

        targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
        targets_x = targets_x.detach()
    return targets_x


class SemiLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self):
        super(SemiLoss, self).__init__()

    def linear_rampup(self, lambda_u, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return lambda_u * float(current)

    def forward(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.linear_rampup(lambda_u, epoch, warm_up)

def dividemix_train_step(args, net, forward, variable_dict, criterion, batch_size, batch_idx, num_iter, epoch, device):
    # Unpack variables
    inputs_x1, inputs_x2, inputs_u1, inputs_u2 = variable_dict['inputs_x1'], variable_dict['inputs_x2'], variable_dict['inputs_u1'], variable_dict['inputs_u2']
    targets_x, targets_u = variable_dict['targets_x'], variable_dict['targets_u']

    # mixmatch
    l = np.random.beta(args.alpha, args.alpha)
    l = max(l, 1 - l)

    all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
    all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    if forward == 'dm':
        logits = net(mixed_input, forward_pass=forward)
    elif forward == 'pcl':
        _, logits = net(mixed_input, forward_pass=forward)
    else:
        raise NotImplementedError
    logits_x = logits[:batch_size * 2]
    logits_u = logits[batch_size * 2:]

    Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                             args.lambda_u, epoch + batch_idx / num_iter, args.warm_up)

    # regularization
    prior = torch.ones(args.num_classes) / args.num_classes
    prior = prior.to(device)
    pred_mean = torch.softmax(logits, dim=1).mean(0)
    penalty = torch.sum(prior * torch.log(prior / pred_mean))

    loss = Lx + lamb * Lu + penalty
    return loss