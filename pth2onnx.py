import numpy as np
import torch
import os
import sys
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.net_utils import load_network
import cv2
from lanedet.datasets.process import Process

# Định nghĩa class wrapper cho ONNX export
class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ONNXExportWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        input_dict = {'img': x}
        output_dict = self.model(input_dict)
        cls_output = output_dict['cls']  # Giữ logits thô
        return cls_output

# Hàm xử lý ảnh đầu vào
def preprocess_image(img_path, cfg):
    print(f"Preprocessing image for conversion: {img_path}")
    ori_img = cv2.imread(img_path)
    h, w = ori_img.shape[0], ori_img.shape[1]
    ori_img = cv2.resize(ori_img, (1640, 590))
    img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    processes = Process(cfg.val_process, cfg)
    data = processes(data)

    # Đảm bảo tensor PyTorch có batch dimension
    data['img'] = data['img'].unsqueeze(0)
    return data['img'], h, w

# Hàm chuyển đổi sang ONNX
def convert(model, config_path, model_path, test_image_path, accuracy, size):
    print('Starting conversion...')

    # Load config
    cfg = Config.fromfile(config_path)
    cfg.batch_size = 1

    # Sử dụng ảnh thực tế để kiểm tra đầu ra
    images, h, w = preprocess_image(test_image_path, cfg)
    images = images.cuda()

    model_wrapped = ONNXExportWrapper(model)
    model_wrapped.eval()

    # Kiểm tra trạng thái mô hình
    print("Checking model state...")
    for name, param in model_wrapped.named_parameters():
        print(f"Parameter {name}: mean={param.mean().item()}, std={param.std().item()}")

    # Kiểm tra đầu ra của mô hình PyTorch trước khi chuyển đổi
    with torch.no_grad():
        output = model_wrapped(images)
    print("Wrapped output shape:", output.shape)
    print("Sample output values (PyTorch):", output[0, :5, :5, :].cpu().numpy())

    onnx_path = model_path[:-4] + "_fp16.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model_wrapped,
            images,
            onnx_path,
            verbose=False,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['cls'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'cls': {0: 'batch_size'}
            }
        )
    print("Export ONNX successful. Model is saved at", onnx_path)

    if accuracy == 'fp16':
        onnx_model = onnxmltools.utils.load_model(onnx_path)
        onnx_model = convert_float_to_float16(onnx_model)
        onnx_half_path = model_path[:-4] + "_fp16.onnx"
        onnxmltools.utils.save_model(onnx_model, onnx_half_path)
        print("FP16 model is saved at", onnx_half_path)

# Thiết lập các tham số trực tiếp thay vì dùng argparse
torch.backends.cudnn.benchmark = True
config_path = '/root/lanedet/configs/ufld/resnet18_culane.py'
model_path = '/root/lanedet/ufld_r18_culane.pth'
test_image_path = '/root/lanedet/images/00000.jpg'
accuracy = 'fp32'  # hoặc 'fp16' tùy ý
size = (800, 288)

# Load config và model
cfg = Config.fromfile(config_path)
cfg.batch_size = 1
net = build_net(cfg)
net = torch.nn.parallel.DataParallel(net, device_ids=range(1)).cuda()
load_network(net, model_path)
net.eval()
net = net.module

# Chạy conversion
convert(net, config_path, model_path, test_image_path, accuracy, size)