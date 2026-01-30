import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
import time
import torch
from  alphabets import plateName,plate_chr
#from plateNet import myNet_ocr
from easydict import EasyDict as edict
import yaml
from lib.dataset import get_dataset
from torch.utils.data import DataLoader
import lib.utils.utils as utils
from tqdm import tqdm
from torch.ao.quantization.fx.graph_module import ObservedGraphModule

plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128]

import torch.nn as nn
import torch.nn.functional as F

class myNet_ocr(nn.Module):
    def __init__(self,cfg=None,num_classes=78,export=False):
        super(myNet_ocr, self).__init__()
        #if cfg is None:
            #cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256]
            # cfg =[32,32,'M',64,64,'M',128,128,'M',256,256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        # self.loc =  nn.MaxPool2d((2, 2), (5, 1), (0, 1),ceil_mode=True)
        # self.loc =  nn.AvgPool2d((2, 2), (5, 2), (0, 1),ceil_mode=False)
        self.loc =  nn.MaxPool2d((5, 2), (1, 1),(0,1),ceil_mode=False)
        self.newCnn=nn.Conv2d(cfg[-1],num_classes,1,1)
        # self.newBn=nn.BatchNorm2d(num_classes)
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        conv2d =nn.Conv2d(in_channels, cfg[0], kernel_size=5,stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[0]), nn.ReLU(inplace=True)]
        in_channels = cfg[0]

        conv2d = nn.Conv2d(in_channels, cfg[1], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[1]), nn.ReLU(inplace=True)]
        in_channels = cfg[1]

        conv2d = nn.Conv2d(in_channels, cfg[2], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[2]), nn.ReLU(inplace=True)]
        in_channels = cfg[2]

        conv2d = nn.Conv2d(in_channels, cfg[3], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[3]), nn.ReLU(inplace=True)]
        in_channels = cfg[3]

        layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]

        conv2d = nn.Conv2d(in_channels, cfg[5], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[5]), nn.ReLU(inplace=True)]
        in_channels = cfg[5]

        conv2d = nn.Conv2d(in_channels, cfg[6], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[6]), nn.ReLU(inplace=True)]
        in_channels = cfg[6]

        layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]

        conv2d = nn.Conv2d(in_channels, cfg[8], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[8]), nn.ReLU(inplace=True)]
        in_channels = cfg[8]

        conv2d = nn.Conv2d(in_channels, cfg[9], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[9]), nn.ReLU(inplace=True)]
        in_channels = cfg[9]

        layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]

        conv2d = nn.Conv2d(in_channels, cfg[11], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[11]), nn.ReLU(inplace=True)]
        in_channels = cfg[11]

        conv2d = nn.Conv2d(in_channels, cfg[12], kernel_size=3, padding=(1,1),stride =1)
        layers += [conv2d, nn.BatchNorm2d(cfg[12]), nn.ReLU(inplace=True)]
        in_channels = cfg[12]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x=self.loc(x)
        #print(x.shape)
        x=self.newCnn(x)
        # x=self.newBn(x)
        conv = x.squeeze(2) # b *512 * width
        conv = conv.transpose(2,1)  # [w, b, c]
        return conv

def calib_quant_model(model, calib_dataloader):
    assert isinstance(
        model, ObservedGraphModule
    ), "model must be a perpared fx ObservedGraphModule."
    model.eval()
    with torch.inference_mode():
        for inputs, labels in tqdm(calib_dataloader):
            model(inputs)
    print("calib done.")

if __name__=='__main__':
    import os
    with open('lib/config/360CC_config.yaml', 'r') as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.safe_load(f)
            config = edict(config)
    config.DATASET.ALPHABETS = plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.HEIGHT=48
    config.WIDTH = 168

    val_dataset = get_dataset(config)(config,input_w=config.WIDTH,input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    criterion = torch.nn.CTCLoss()
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    fp32_model = myNet_ocr(num_classes=len(plate_chr),cfg=cfg,export=True).eval()
    checkpoint = torch.load('weights/plate_rec.pth', map_location='cpu')
    fp32_model.load_state_dict(checkpoint['state_dict'])

    model = copy.deepcopy(fp32_model)
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    model_prepared = prepare_fx(model, qconfig_dict,example_inputs=None)
    calib_quant_model(model_prepared,val_loader)
    model_quantized = convert_fx(model_prepared)#,is_reference=True)
    model_quantized.eval()
    print(model_quantized)
    
    input = torch.randn(1,3,48,168)
    dynamic=False
    torch.onnx.export(model_quantized,input,'plate_rec_quant.onnx',
                    input_names=["images"],output_names=["output"],
                    verbose=True,
                    do_constant_folding=True,
                    opset_version=11,
                    dynamic_axes={'images': {0: 'batch'},
                                'output': {0: 'batch'}
                                } if dynamic else None)
    print(f"convert completed,save to {'plate_rec_quant.onnx'}")