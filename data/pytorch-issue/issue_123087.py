import torch
import torch.nn as nn

class DataCov(nn.Module):
    def __init__(self):
        super(DataCov, self).__init__()

        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        return self.transform(x1)


class QANet(nn.Module):
    def __init__(self, num_classes=1):
        super(QANet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.bn = nn.Sequential(
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1)
        )

        self.fc1_3_0 = nn.Sequential(
            nn.Linear(375, 375),
            nn.GELU()
        )

        self.fc1_3_1 = nn.Sequential(
            nn.Linear(375, 375),
            nn.GELU()
        )

        self.fc1_3_2 = nn.Sequential(
            nn.Linear(375, 375),
            nn.GELU()
        )

        self.fc1_3_3 = nn.Sequential(
            nn.Linear(375, 375),
            nn.GELU()
        )

        self.fc1_3_out = nn.Sequential(
            nn.Linear(375, 1),
            nn.GELU()
        )

        self.fc1_2_0 = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.fc1_2_1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.fc1_2_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.fc1_2_3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.fc1_2_out = nn.Sequential(
            nn.Linear(64, 1),
            nn.GELU()
        )

        self.fc1_1_0 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.fc1_1_1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.fc1_1_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.fc1_1_3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.fc1_1_out = nn.Sequential(
            nn.Linear(256, 1),
            nn.GELU()
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        x = torch.cat((x1, x2), dim=1)
        del x1, x2
        x = self.fc1_3_0(x)
        x = self.fc1_3_1(x)
        x = self.fc1_3_2(x)
        x = self.fc1_3_3(x)
        x = self.fc1_3_out(x)
        x = x.view(x.size(0), 256, 64)
        x = self.fc1_2_0(x)
        x = self.fc1_2_1(x)
        x = self.fc1_2_2(x)
        x = self.fc1_2_3(x)
        x = self.fc1_2_out(x)
        x = x.view(x.size(0), 256)
        x = self.fc1_1_0(x)
        x = self.fc1_1_1(x)
        x = self.fc1_1_2(x)
        x = self.fc1_1_3(x)
        x = self.fc1_1_out(x)
        return x


def export_datacov_onnx(pth_path, path):
    state_dict = torch.load(pth_path, map_location='cpu')
    model = QANet().to(torch.float32)
    set_compile = True
    if set_compile:
        state_dict_fix = {}
        for k, v in state_dict.items():
            tmp_k = k[k.find(".") + 1:]
            state_dict_fix[tmp_k] = v
        model.load_state_dict(state_dict_fix, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    model_cov = DataCov()
    model_cov.eval()
    src_wav = torch.randn((1, 1, 48000 * 12), requires_grad=True)
    dst_wav = torch.randn((1, 1, 48000 * 12), requires_grad=True)
    input_names = ["sample_data_cov", "src_data_cov"]        # 导出的ONNX模型输入节点名称
    output_names = ["ans"]      # 导出的ONNX模型输出节点名称
    x_1 = model_cov(dst_wav)
    x_2 = model_cov(src_wav)
    args = (x_1, x_2)