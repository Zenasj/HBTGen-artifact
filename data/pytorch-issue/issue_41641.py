import torch
import torch.nn as nn

class PReLU_Quantized(nn.Module):
    def __init__(self, prelu_object):
        super().__init__()
        self.prelu_weight = prelu_object.weight
        self.weight = self.prelu_weight
        self.quantized_op = nn.quantized.FloatFunctional()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, inputs):
        # inputs = max(0, inputs) + alpha * min(0, inputs) 
        # this is how we do it 
        # pos = torch.relu(inputs)
        # neg = -alpha * torch.relu(-inputs)
        # res3 = pos + neg
        self.weight = self.quant(self.weight)
        weight_min_res = self.quantized_op.mul(-self.weight, torch.relu(-inputs))
        inputs = self.quantized_op.add(torch.relu(inputs), weight_min_res)
        inputs = self.dequant(inputs)
        self.weight = self.dequant(self.weight)
        return inputs

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mult_xy = nn.quantized.FloatFunctional()

        self.fc = nn.Sequential(
                                nn.Linear(channel, channel // reduction),
                                nn.PReLU(),
                                # nn.ReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid()
                                )
        self.fc1 = self.fc[0]
        self.prelu = self.fc[1]
        self.fc2 = self.fc[2]
        self.sigmoid = self.fc[3]
        self.prelu_q = PReLU_Quantized(self.prelu)

    def forward(self, x):
        print(f'<inside se forward:>')
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc1(y)
        print(f'X: {y}')
        # unlike in other modules, this instance of prelu_q causes this issue, this is not the case in other modules!
        y = self.prelu_q(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        print('--------------------------')
        # out = x*y 
        out = self.mult_xy.mul(x, y)
        return out

class PReLU_Quantized(nn.Module):
    def __init__(self, prelu_object):
        super().__init__()
        self.prelu_weight = prelu_object.weight
        self.weight = self.prelu_weight
        self.quantized_op = nn.quantized.FloatFunctional()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, inputs):
        # inputs = max(0, inputs) + alpha * min(0, inputs) 
        # this is how we do it 
        # pos = torch.relu(inputs)
        # neg = -alpha * torch.relu(-inputs)
        # res3 = pos + neg
        self.weight = self.quant(self.weight)
        weight_min_res = self.quantized_op.mul(-self.weight, torch.relu(-inputs))
        inputs = self.quantized_op.add(torch.relu(inputs), weight_min_res)
        inputs = self.dequant(inputs)
        self.weight = self.dequant(self.weight)
        return inputs

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)
        out = self.add_relu.add_relu(out, residual)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        
        self.skip_add_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # out += residual
        # out = self.relu(out)
        out = self.skip_add_relu.add_relu(out, residual)
        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mult_xy = nn.quantized.FloatFunctional()

        self.fc = nn.Sequential(
                                nn.Linear(channel, channel // reduction),
                                nn.PReLU(),
                                # nn.ReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid()
                                )
        self.fc1 = self.fc[0]
        self.prelu = self.fc[1]
        self.fc2 = self.fc[2]
        self.sigmoid = self.fc[3]
        self.prelu_q = PReLU_Quantized(self.prelu)

    def forward(self, x):
        print(f'<inside se forward:>')
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc1(y)
        print(f'X: {y}')
        y = self.prelu_q(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        print('--------------------------')
        # out = x*y 
        out = self.mult_xy.mul(x, y)
        return out

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        # self.prelu = nn.ReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

        self.add_residual_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)

        # out = self.prelu(out)
        out = self.prelu_q(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.prelu(out)

        # we may need to change prelu into relu and this, instead of add, use add_relu here
        out = self.add_residual_relu.add_relu(out, residual)
        # out = self.prelu(out)
        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1'],# 'prelu'],
                            ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        # self.prelu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        # self.bn2_q = BatchNorm2d_Quantized(self.bn2)
        # self.bn3_q = BatchNorm1d_Quantized(self.bn3)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)

        # x = self.prelu(x)
        x = self.prelu_q(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        # x = self.bn2_q(x)
        x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        # x = self.bn3_q(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, [['conv1', 'bn1'],# 'prelu'],
                            # ['bn2'],  ['bn3']
                            ], inplace=True)
        for m in self.modules():
            # print(m)
            if type(m) == Bottleneck or type(m) == BasicBlock or type(m) == IRBlock:
                m.fuse_model()

def resnet18(pretrained, use_se, **kwargs):
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def quantize_test():
    m = resnet18()
    num_calibration_batches = 10 
    saved_model_dir = 'data'
    scripted_quantized_model_file  = 'model_quantized_jit.pth'
    # you can download this from here : https://github.com/foamliu/InsightFace-v2/releases/tag/v1.0 
    # but you may skip laoding and go straight for the quantization to see the issue
    model_checkpoint_path = 'BEST_checkpoint_r18.tar'
    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
    model = resnet18(pretrained=False, use_se=checkpoint['use_se'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.fuse_model()
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    print(f'Model after quantization(converted-prepared): {model}')
    
    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', model.conv1)

    # Calibrate with the training set
    evaluate(model, dtloader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n', model.conv1)

    script = torch.jit.script(model)
    path_tosave = os.path.join(os.path.dirname(os.path.abspath(__file__)), saved_model_dir , scripted_quantized_model_file)
    print(f'path to save: {path_tosave}')
    with open(path_tosave,'wb') as f:
        torch.save(model.state_dict(), f)