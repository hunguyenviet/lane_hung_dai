import numpy as np
import torch
import cv2
import os
import json
import random
from tqdm import tqdm
import time
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        h, w = ori_img.shape[0], ori_img.shape[1]
        ori_img = cv2.resize(ori_img, (1640, 590))
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img': ori_img, 'h': h, 'w': w})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show)
        img = data['ori_img']
        h = data['h']
        w = data['w']
        img = cv2.resize(img, (w, h))
        return img, lanes

    def run(self, frame):
        data = self.preprocess(frame)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            frame, lanes = self.show(data)
            return frame, lanes
        return frame, []

def save_lane_coordinates(lanes, txt_path):
    """Save lane coordinates in CULane format"""
    with open(txt_path, 'w') as f:
        for lane in lanes:
            points = []
            for x, y in lane:
                if x >= 0 and y >= 0:
                    points.append(f"{int(x)} {int(y)}")
            if points:
                f.write(" ".join(points) + "\n")

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to serializable formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj

def create_annotation_json(lanes, json_path):
    """Create JSON annotation file from lane data"""
    annotations = []
    for i, lane in enumerate(lanes, start=1):
        if len(lane) >= 2:
            # Convert numpy arrays to lists
            start_point = [int(round(x)) for x in lane[0]]
            end_point = [int(round(x)) for x in lane[-1]]

            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            annotations.append({
                "id": i,
                "annotations": [
                    [
                        "line",
                        start_point,
                        end_point,
                        color,
                        2
                    ]
                ],
                "gray_value": random.randint(0, 255)
            })

    json_data = {
        "annotations": annotations,
        "lane_characteristics": ["Normal"]
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4, default=convert_to_serializable)

def process_video(config, show, output_dir, input_video, model):
    cfg = Config.fromfile(config)
    cfg.show = show
    cfg.savedir = output_dir
    cfg.load_from = model
    detect = Detect(cfg)

    # Create output directories
    frames_dir = os.path.join(output_dir, 'frames')
    gt_image_dir = os.path.join(output_dir, 'gt_image')
    annotations_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    vid = cv2.VideoCapture(input_video)
    if not vid.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    save_interval = 30

    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while True:
            start = time.time()
            ret, frame = vid.read()
            if not ret:
                break

            try:
                # Save original frame (resized to 1640x590)
                resized_frame = cv2.resize(frame, (1640, 590))

                # Process frame and get lanes
                processed_frame, lanes = detect.run(frame)

                # Save ground truth data every save_interval frames
                if frame_idx % save_interval == 0:
                    frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, resized_frame)
                    # Save gt image
                    gt_path = os.path.join(gt_image_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(gt_path, resized_frame)

                    # Save lane coordinates
                    txt_path = os.path.join(gt_image_dir, f"{frame_idx:06d}.lines.txt")
                    save_lane_coordinates(lanes, txt_path)

                    # Create and save JSON annotation
                    json_path = os.path.join(annotations_dir, f"{frame_idx:06d}.json")
                    create_annotation_json(lanes, json_path)

                frame_idx += 1
                pbar.update(1)

            except Exception as e:
                print(f"\nError processing frame {frame_idx}: {str(e)}")
                continue

    vid.release()
    print(f"\nProcessing complete. Results saved in:")
    print(f"- Original frames: {frames_dir}")
    print(f"- Ground truth images and labels: {gt_image_dir}")
    print(f"- JSON annotations: {annotations_dir}")
    print(f"- Processed video: {os.path.join(output_dir, 'output.mp4')}")

if __name__ == '__main__':
    # Parameters
    config = '/root/lanedet/configs/ufld/resnet18_culane.py'
    show = False
    output_dir = 'lane_label_tool'
    input_video = '/root/lanedet/LOCK5138.avi'
    model = '/root/lanedet/work_dirs/CULane/20250416_003849_lr_1e-02_b_8/ckpt/best.pth'

    # Run processing
    process_video(config, show, output_dir, input_video, model)