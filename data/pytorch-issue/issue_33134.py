import torch
import torch.nn as nn
from torch.nn import BatchNorm3d

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv3d(1, 64, 3, padding=1),
            BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.convM1 = nn.Sequential(
            # conv1
            nn.Conv3d(1, 64, 3, padding=1),
            BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.d1 = nn.Conv3d(64,1,1)
        self.e1 = nn.Conv3d(64,6,1)
    def forward(self, x):
        conv1 = self.conv1(x)
        e1 = self.e1(conv1)
        e1_sum = torch.sum(e1, 1).unsqueeze(1)
        convM1 = self.convM1(e1_sum)
        d1 = self.d1(convM1)
        return {'mask':[d1],
                'exp':[e1]}

class DiceLoss(nn.Module):
    '''soft dice loss'''
    '''Computes the Dice Loss (dice) as described in https://arxiv.org/pdf/1707.03237.pdf'''

    def __init__(self, reduce_axes=[1, 2, 3, 4], smooth=1.0, epsilon=1e-7, final_reduction=torch.mean):
        super(DiceLoss, self).__init__()
        self.reduce_axes = reduce_axes
        self.smooth = smooth
        self.eps = epsilon
        self.final_reduction = final_reduction

    def forward(self, predictions, labels):
        """
        Simple functional form of dice loss
        """
        predictions = nn.Sigmoid()(predictions)
        dice_loss = 1.0 - self.dice_fn(predictions, labels, self.reduce_axes, self.smooth, self.eps)

        if self.final_reduction:
            dice_loss = self.final_reduction(dice_loss)

        return dice_loss

    def dice_fn(self, predictions, labels, reduce_axes, smooth, eps):
        """
        Can accept a soft count relaxation or a true binary input
        Uses the squared denominator form as Milletari demonstrated that its
        gradient behaves better
        """
        intersection = labels * predictions

        intersection = intersection.sum(dim=reduce_axes)
        labels = (labels * labels).sum(dim=reduce_axes)
        predictions = (predictions * predictions).sum(dim=reduce_axes)

        dice = (2.0 * intersection + smooth) / \
               (labels + predictions + smooth + eps)

        return dice

class DicePlusBCE(nn.Module):
    def __init__(self, reduce_axes=[1, 2, 3, 4], smooth=1.0, epsilon=1e-7, final_reduction=torch.mean, pos_weight=None):
        super(DicePlusBCE, self).__init__()
        self.reduce_axes = reduce_axes
        self.smooth = smooth
        self.eps = epsilon
        self.final_reduction = final_reduction
        self.pos_weight = pos_weight
        self.dice_loss = DiceLoss(reduce_axes, smooth, epsilon, final_reduction)
        self.focal_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        dice_loss = self.dice_loss(predictions, labels)
        bce_loss = self.focal_loss(predictions, labels)
        return dice_loss + bce_loss

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        # target_weight = torch.ones_like(target)
        # target_weight[target > 0] = 100
        # target_weight = target_weight.view(target.shape[0], 6, -1)
        # import pdb;pdb.set_trace()
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred,
                    heatmap_gt
                )
            else:
                loss += 1 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

model = Model()
model_parallel = torch.nn.DataParallel(model).cuda()
mask_loss = DicePlusBCE()
exp_loss = JointsMSELoss()
x = torch.randn(2,1,48,256,256)
x = torch.autograd.Variable(x).cuda()
gt = torch.autograd.Variable(torch.zeros_like(x)).cuda()
exp_gt = torch.autograd.Variable(torch.zeros(2,6,48,256,256)).cuda()
outs = model_parallel(x)

loss_m = mask_loss(outs['mask'][-1], gt)
loss_exp = exp_loss(outs['exp'][-1], exp_gt)
loss = loss_exp+loss_m
loss.backward()