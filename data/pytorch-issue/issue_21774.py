import torch
import torch.nn as nn
import numpy as np

class YOLOv3(nn.Module):
    def __init__(self,
                 img_width,
                 img_height,
                 anchors,
                 anch_mask,
                 n_classes,
                 ignore_thre=0.7):
        super(YOLOv3, self).__init__()

        self.module_list = create_yolov3_modules(img_width,
                                                 img_height,
                                                 anchors,
                                                 anch_mask,
                                                 n_classes,
                                                 ignore_thre)
        self.length = len(self.module_list)

    def forward(self, x, targets=None):
        train = targets is not None
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'detects',
                                           'nondetects', 'class'],
                                          loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)

class YOLOLayer(nn.Module):
    def __init__(self,
                 anchors,
                 anch_mask,
                 n_classes,
                 layer_no,
                 in_ch,
                 ignore_thre=0.45,
                 nms_thre=0.7,
                 net_map=""):

        super(YOLOLayer, self).__init__()
        self.net_map = net_map
        strides = [32, 16, 8]  # fixed
        self.anchors = anchors
        self.anch_mask = anch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = n_classes
        self.ignore_thre = ignore_thre
        self.nms_thre = nms_thre
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = strides[layer_no]
        # As the image is scaled by stride factor, the anchors should as well
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors *
                              (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=1)


    def forward(self, xin, labels=None):
        output = self.conv(xin)
        # output shape: BatchSize, NumAnchors, Channels, H, W
        batchsize = output.shape[0]
        hout = output.shape[2]
        wout = output.shape[3]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, hout, wout)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = torch.zeros(output.shape[:4], dtype=torch.float32).cuda()
        ar = torch.arange(wout, dtype=torch.float32).cuda()
        x_shift = x_shift + ar
        x_shift = x_shift
        
        y_shift = torch.zeros(output.shape[:4], dtype=torch.float32).cuda()
        ary = torch.arange(hout, dtype=torch.float32).cuda()
        y_p = y_shift.permute([0,1,3,2])
        y_p = y_p + ary
        y_shift = y_p.permute([0,1,3,2])

        masked_anchors = torch.tensor(self.masked_anchors, dtype=torch.float32).cuda()

        w_anchors = torch.zeros(output.shape[:4],
                                dtype=torch.float32).cuda()
        w_p = w_anchors.permute([0,3,2,1])
        w_p = w_p + masked_anchors[:,0]
        w_anchors = w_p.permute([0,3,2,1])

        h_anchors = torch.zeros(output.shape[:4],
                                dtype=torch.float32).cuda()
        h_p = h_anchors.permute([0,3,2,1])
        h_p = h_p + masked_anchors[:,1]
        h_anchors = h_p.permute([0,3,2,1])


        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.sqrt(torch.exp(pred[..., 2]) *
                                  w_anchors)  # Already scaled as anchors are
        pred[..., 3] = torch.sqrt(torch.exp(pred[..., 3]) *
                                  h_anchors)  # Already scaled as anchors are

        if labels is None:  # not training
            pred[..., 2] = pred[..., 2]**2
            pred[..., 3] = pred[..., 3]**2
            pred[..., :4] *= self.stride
            return pred.view(batchsize, -1, n_ch).cpu().data

...

model.eval()
imgs_temp = imgs[0].view(1, imgs.shape[1], imgs.shape[2], imgs.shape[3])
traced_script_module = torch.jit.trace(model, imgs_temp)
output = traced_script_module(imgs_temp)
traced_script_module.save("./testmod.pt")