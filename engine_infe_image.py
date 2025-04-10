import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from lanedet.datasets.process import Process
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.core.lane import Lane
import torch
import onnxruntime as ort
import os
import time

# Paths to engine file, image, and config
engine_file_path = '/root/lanedet/ufld_r18_culane_fp16.engine'
image_path = '/root/lanedet/images/00000.jpg'  # Replace with your image path
config_path = '/root/lanedet/configs/ufld/resnet18_culane.py'
output_image_path = '/root/lanedet/vis/engine.jpg'  # Path to save the output image

# Load config
cfg = Config.fromfile(config_path)

# Create TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Preprocess image
def preprocess_image(ori_img, cfg):
    h, w = ori_img.shape[0], ori_img.shape[1]
    ori_img = cv2.resize(ori_img, (1640, 590))
    img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    processes = Process(cfg.val_process, cfg)
    data = processes(data)

    # Convert to numpy format
    img_numpy = data['img'].numpy()
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)

    return img_numpy, ori_img, h, w

# Run inference with TensorRT engine
def inference_tensorrt(engine, img_numpy):
    context = engine.create_execution_context()

    input_name = "input"
    output_name = "cls"

    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

    # Input
    input_shape = engine.get_tensor_shape(input_name)
    input_size = trt.volume(input_shape)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_host_mem = cuda.pagelocked_empty(input_size, input_dtype)
    input_device_mem = cuda.mem_alloc(input_host_mem.nbytes)
    inputs.append({'host': input_host_mem, 'device': input_device_mem})
    bindings.append(int(input_device_mem))
    context.set_tensor_address(input_name, int(input_device_mem))

    # Output
    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape)
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output_host_mem = cuda.pagelocked_empty(output_size, output_dtype)
    output_device_mem = cuda.mem_alloc(output_host_mem.nbytes)
    outputs.append({'host': output_host_mem, 'device': output_device_mem})
    bindings.append(int(output_device_mem))
    context.set_tensor_address(output_name, int(output_device_mem))

    # Transfer input data to GPU
    inputs[0]['host'][:] = img_numpy.ravel()
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Transfer output data from GPU to CPU
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    cls_output = outputs[0]['host'].reshape(1, 201, 18, 4)
    return cls_output

# Process and draw lanes
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

# Build and load model
from lanedet.models.registry import build_net
from lanedet.utils.net_utils import load_network

model = build_net(cfg)
model = torch.nn.parallel.DataParallel(model, device_ids=range(1)).cuda()
load_network(model, '/root/lanedet/ufld_r18_culane.pth')
model.eval()

# Load engine
engine = load_engine(engine_file_path)

# Read and process the image
ori_img = cv2.imread(image_path)
if ori_img is None:
    raise ValueError(f"Could not load image from {image_path}")

start = time.time()
img_numpy, ori_img, h, w = preprocess_image(ori_img, cfg)
cls_output = inference_tensorrt(engine, img_numpy)
img_with_lanes = process_and_draw_lanes(cls_output, model, ori_img, cfg, h, w)
end = time.time()

# Calculate FPS
fps = 1 / (end - start)
cv2.putText(img_with_lanes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Save the output image
cv2.imwrite(output_image_path, img_with_lanes)
print(f"Output image saved to {output_image_path}")

# Display the result
cv2.imshow('Lane Detection', img_with_lanes)
cv2.waitKey(0)  # Wait for any key press to close the window
cv2.destroyAllWindows()