# %%writefile /kaggle/working/lanedet/lanedet/models/backbones/efficientnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from lanedet.models.registry import BACKBONES

# URL pretrained weights cho EfficientNet B0-B7
model_urls = {
    'EfficientNetB0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f13e928.pth',
    'EfficientNetB1': 'https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287c5.pth',
    'EfficientNetB2': 'https://download.pytorch.org/models/efficientnet_b2_rwightman-08ebbf45.pth',
    'EfficientNetB3': 'https://download.pytorch.org/models/efficientnet_b3_rwightman-176b4795.pth',
    'EfficientNetB4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-8868b1c5.pth',
    'EfficientNetB5': 'https://download.pytorch.org/models/efficientnet_b5_rwightman-989b33f9.pth',
    'EfficientNetB6': 'https://download.pytorch.org/models/efficientnet_b6_rwightman-03c36fc4.pth',
    'EfficientNetB7': 'https://download.pytorch.org/models/efficientnet_b7_rwightman-3ddd14e1.pth',
}

# Hàm phụ để làm tròn số kênh sao cho chia hết cho divisor
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_channels = _make_divisible(in_channels // reduction, 8)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return x * out

# MBConv block (Mobile Inverted Bottleneck Convolution)
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, dilation=1, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = _make_divisible(in_channels * expand_ratio, 8)

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        # Depthwise convolution
        padding = (kernel_size - 1) // 2 * dilation
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                padding=padding, groups=hidden_dim, dilation=dilation, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())

        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock(hidden_dim, reduction=int(1 / se_ratio)))

        # Projection phase
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Cấu hình cơ bản cho EfficientNet (B0) và scaling factors cho B1-B7
efficientnet_configs = {
    # [width_mult, depth_mult, resolution]
    'EfficientNetB0': (1.0, 1.0, 224),
    'EfficientNetB1': (1.0, 1.1, 240),
    'EfficientNetB2': (1.1, 1.2, 260),
    'EfficientNetB3': (1.2, 1.4, 300),
    'EfficientNetB4': (1.4, 1.8, 380),
    'EfficientNetB5': (1.6, 2.2, 456),
    'EfficientNetB6': (1.8, 2.6, 528),
    'EfficientNetB7': (2.0, 3.1, 600),
}

# Cấu hình layer cho EfficientNet B0 (sẽ được scale cho các phiên bản khác)
base_config = [
    # [expand_ratio, out_channels, num_layers, stride, kernel_size]
    [1, 16, 1, 1, 3],  # Stage 1
    [6, 24, 2, 2, 3],  # Stage 2
    [6, 40, 2, 2, 5],  # Stage 3
    [6, 80, 3, 2, 3],  # Stage 4
    [6, 112, 3, 1, 5], # Stage 5
    [6, 192, 4, 2, 5], # Stage 6
    [6, 320, 1, 1, 3], # Stage 7
]

class EfficientNetInner(nn.Module):
    def __init__(self, version='EfficientNetB0', width_mult=1.0, depth_mult=1.0):
        super(EfficientNetInner, self).__init__()
        self.version = version
        width_mult, depth_mult, _ = efficientnet_configs[version]

        # Đầu vào ban đầu
        in_channels = _make_divisible(32 * width_mult, 8)
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.swish = Swish()

        # Xây dựng các stage
        self.stages = nn.ModuleList()
        current_channels = in_channels
        for expand_ratio, out_channels, num_layers, stride, kernel_size in base_config:
            out_channels = _make_divisible(out_channels * width_mult, 8)
            num_layers = max(1, int(round(num_layers * depth_mult)))
            layers = []
            for i in range(num_layers):
                s = stride if i == 0 else 1
                layers.append(MBConv(current_channels, out_channels, kernel_size, s, expand_ratio, se_ratio=0.25))
                current_channels = out_channels
            self.stages.append(nn.Sequential(*layers))

        # Đầu ra cuối cùng
        last_channels = _make_divisible(1280 * width_mult, 8)
        self.conv_last = nn.Conv2d(current_channels, last_channels, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(last_channels)

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish(x)

        out_layers = []
        for stage in self.stages:
            x = stage(x)
            out_layers.append(x)

        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.swish(x)
        out_layers[-1] = x  # Ghi đè layer cuối với đầu ra sau conv_last

        return out_layers

@BACKBONES.register_module
class EfficientNet(nn.Module):
    def __init__(self, 
                 net='EfficientNetB0',
                 pretrained=True,
                 replace_stride_with_dilation=[False, False, False],  # Không dùng, để tương thích với ResNet
                 out_conv=False,
                 fea_stride=8,  # Không dùng trực tiếp, để tương thích
                 out_channel=128,  # Mặc định, có thể thay đổi trong config
                 in_channels=[32, 16, 24, 40, 80, 112, 192, 320],  # Số kênh mặc định của B0 trước khi scale
                 cfg=None):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        width_mult, depth_mult, _ = efficientnet_configs[net]
        self.model = EfficientNetInner(net, width_mult, depth_mult)
        
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[net])
            self.model.load_state_dict(state_dict, strict=False)

        self.out = None
        if out_conv:
            out_channel = self.model.stages[-1][-1].out_channels  # Lấy số kênh từ layer cuối
            self.out = nn.Conv2d(out_channel, cfg.featuremap_out_channel, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x

# Hàm tạo các phiên bản EfficientNet B0-B7
def _efficientnet(version, pretrained=False, progress=True, **kwargs):
    model = EfficientNet(net=version, pretrained=pretrained, **kwargs)
    return model

def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB0', pretrained, progress, **kwargs)

def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB1', pretrained, progress, **kwargs)

def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB2', pretrained, progress, **kwargs)

def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB3', pretrained, progress, **kwargs)

def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB4', pretrained, progress, **kwargs)

def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB5', pretrained, progress, **kwargs)

def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB6', pretrained, progress, **kwargs)

def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    return _efficientnet('EfficientNetB7', pretrained, progress, **kwargs)