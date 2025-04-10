import numpy as np
import torch
import cv2
import onnxruntime as ort
import os
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane
import time


# Đường dẫn đến các file
onnx_model_path = '/root/lanedet/ufld_r18_culane_fp16.onnx'
config_path = '/root/lanedet/configs/ufld/resnet18_culane.py'
video_path = '/root/lanedet/IMG_5132.MOV'  # Đường dẫn video đầu vào
output_video_path = './vis/output_video_onnx.mp4'  # Đường dẫn lưu video đầu ra

# Tạo thư mục lưu video đầu ra
os.makedirs('./vis', exist_ok=True)

# Tải config
cfg = Config.fromfile(config_path)

# Hàm tiền xử lý frame từ video
def preprocess_frame(frame, cfg):
    h, w = frame.shape[0], frame.shape[1]
    ori_frame = cv2.resize(frame, (1640, 590))  # Thay đổi kích thước như ảnh gốc
    img = ori_frame[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    processes = Process(cfg.val_process, cfg)
    data = processes(data)

    # Chuyển sang định dạng numpy cho ONNX
    img_numpy = data['img'].numpy()
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)

    return img_numpy, ori_frame, h, w

# Hàm chạy suy luận với model ONNX
def inference_onnx(session, img_numpy, model):
    # if img_numpy.dtype != np.float16:
    #     img_numpy = img_numpy.astype(np.float16)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    cls_output = session.run([output_name], {input_name: img_numpy})[0]
    cls_output_tensor = torch.from_numpy(cls_output).cuda()
    output_dict = {'cls': cls_output_tensor}
    with torch.no_grad():
        lanes = model.module.get_lanes(output_dict)
    return lanes

# Hàm xử lý và vẽ lane lên frame
def process_and_draw_lanes(lanes, frame, cfg, h, w):
    processed_lanes = []
    for lane_group in lanes:
        for item in lane_group:
            if isinstance(item, Lane):
                coords = item.to_array(cfg)
                # Lọc bỏ các điểm không hợp lệ
                coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < w) &
                                (coords[:, 1] >= 0) & (coords[:, 1] < h)]
                if len(coords) > 1:
                    processed_lanes.append(coords.astype(np.int32))

    frame_with_lanes = frame.copy()
    imshow_lanes(frame_with_lanes, processed_lanes, show=False)
    frame_with_lanes = cv2.resize(frame_with_lanes, (w, h))
    return frame_with_lanes

# Tải model ONNX và khởi tạo mô hình để lấy hàm get_lanes
model = build_net(cfg)  # Cần thiết để sử dụng hàm get_lanes
model = torch.nn.parallel.DataParallel(model, device_ids=range(1)).cuda()
model.eval()
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Mở video đầu vào
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Lấy thông tin video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Khởi tạo video writer để lưu kết quả
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Biến để tính FPS
prev_time = 0

# Xử lý từng frame trong video
while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    # Tiền xử lý frame
    img_numpy, ori_frame, h, w = preprocess_frame(frame, cfg)

    # Chạy suy luận với ONNX
    lanes = inference_onnx(session, img_numpy, model)

    # Vẽ lane lên frame
    frame_with_lanes = process_and_draw_lanes(lanes, ori_frame, cfg, h, w)
    end = time.time()
    fps_real = 1 / (end - start)

    # Hiển thị FPS trên frame
    cv2.putText(frame_with_lanes, f"FPS: {fps_real:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Ghi frame vào video đầu ra
    out.write(frame_with_lanes)

    # Hiển thị frame (tùy chọn, có thể bỏ nếu không cần xem trực tiếp)
    cv2.imshow('Lane Detection', frame_with_lanes)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video with lanes saved at: {output_video_path}")