import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from lanedet.datasets.process import Process
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.core.lane import Lane
import numpy as np
import torch
import cv2
import onnxruntime as ort
import os
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane
import time
# Đường dẫn đến file engine và ảnh kiểm tra
engine_file_path = '/root/lanedet/ufld_r18_culane_fp16.engine'
video_path = '/root/lanedet/IMG_5132.MOV'
config_path = '/root/lanedet/configs/ufld/resnet18_culane.py'

# Tải config
cfg = Config.fromfile(config_path)

# Tạo logger cho TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Tải TensorRT engine
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Tiền xử lý ảnh
def preprocess_image(ori_img, cfg):
    h, w = ori_img.shape[0], ori_img.shape[1]
    ori_img = cv2.resize(ori_img, (1640, 590))
    img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    processes = Process(cfg.val_process, cfg)
    data = processes(data)

    # Chuyển sang định dạng numpy
    img_numpy = data['img'].numpy()
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)

    return img_numpy, ori_img, h, w

# Chạy suy luận với TensorRT engine
def inference_tensorrt(engine, img_numpy):
    # Tạo context
    context = engine.create_execution_context()

    # Tên của đầu vào và đầu ra
    input_name = "input"
    output_name = "cls"

    # Cấp phát bộ nhớ cho đầu vào và đầu ra
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

    # Đầu vào
    input_shape = engine.get_tensor_shape(input_name)
    input_size = trt.volume(input_shape)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_host_mem = cuda.pagelocked_empty(input_size, input_dtype)
    input_device_mem = cuda.mem_alloc(input_host_mem.nbytes)
    inputs.append({'host': input_host_mem, 'device': input_device_mem})
    bindings.append(int(input_device_mem))
    context.set_tensor_address(input_name, int(input_device_mem))

    # Đầu ra
    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape)
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output_host_mem = cuda.pagelocked_empty(output_size, output_dtype)
    output_device_mem = cuda.mem_alloc(output_host_mem.nbytes)
    outputs.append({'host': output_host_mem, 'device': output_device_mem})
    bindings.append(int(output_device_mem))
    context.set_tensor_address(output_name, int(output_device_mem))

    # Chuyển dữ liệu đầu vào lên GPU
    inputs[0]['host'][:] = img_numpy.ravel()
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # Chạy suy luận
    context.execute_async_v3(stream_handle=stream.handle)  # Thay execute_async_v2 bằng execute_async_v3

    # Chuyển dữ liệu đầu ra từ GPU về CPU
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Định dạng lại đầu ra
    cls_output = outputs[0]['host'].reshape(1, 201, 18, 4)
    return cls_output

# Xử lý và vẽ lane
def process_and_draw_lanes(cls_output, model, ori_img, cfg, h, w):
    cls_output_tensor = torch.from_numpy(cls_output).cuda()
    output_dict = {'cls': cls_output_tensor}
    with torch.no_grad():
        lanes = model.module.get_lanes(output_dict)

    processed_lanes = []
    print(f"Processing lanes: {lanes}")
    for lane_group in lanes:
        print(f"lane_group: {lane_group}")
        for i, item in enumerate(lane_group):
            print(f"Item {i}: {item}, Type: {type(item)}")
            if isinstance(item, Lane):
                coords = item.to_array(cfg)
                print(f"Processed lane coords (before filtering): {coords}")
                coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < w) &
                                (coords[:, 1] >= 0) & (coords[:, 1] < h)]
                print(f"Processed lane coords (after filtering): {coords}")
                if len(coords) > 1:
                    processed_lanes.append(coords.astype(np.int32))
            else:
                print(f"Skipping non-Lane item: {item}")

    print(f"Number of lanes to draw: {len(processed_lanes)}")
    for i, lane in enumerate(processed_lanes):
        print(f"Lane {i} coordinates: {lane}")

    img_with_lanes = ori_img.copy()
    imshow_lanes(img_with_lanes, processed_lanes, show=False)
    img_with_lanes = cv2.resize(img_with_lanes, (w, h))
    return img_with_lanes

from lanedet.models.registry import build_net
from lanedet.utils.net_utils import load_network
model = build_net(cfg)
model = torch.nn.parallel.DataParallel(model, device_ids=range(1)).cuda()

load_network(model, '/root/lanedet/ufld_r18_culane.pth')
model.eval()
engine = load_engine(engine_file_path)
cap = cv2.VideoCapture(video_path)
while True:
  start = time.time()
  ret, frame = cap.read()
  if not ret:
    break
  img_numpy, ori_img, h, w = preprocess_image(frame, cfg)
  cls_output = inference_tensorrt(engine, img_numpy)
  img_with_lanes = process_and_draw_lanes(cls_output, model, ori_img, cfg, h, w)
  end = time.time()
  fps = 1 / (end - start)
  cv2.putText(img_with_lanes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('Lane Detection', img_with_lanes)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break