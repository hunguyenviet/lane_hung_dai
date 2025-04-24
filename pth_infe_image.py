import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
from pathlib import Path
from tqdm import tqdm
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)
        print(f"Loaded PyTorch model from: {self.cfg.load_from}")

    def preprocess(self, img_path):
        print(f"Preprocessing image: {img_path}")
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            raise FileNotFoundError(f"Could not load image from {img_path}. Please check the file path.")
        h, w = ori_img.shape[0], ori_img.shape[1]
        ori_img = cv2.resize(ori_img, (1640, 590))
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        print(f"Input shape for PyTorch: {data['img'].shape}")
        data.update({'img_path': img_path, 'ori_img': ori_img, 'h': h, 'w': w})
        return data

    def inference(self, data):
        print("Running PyTorch inference...")
        with torch.no_grad():
            data = self.net(data)
            lanes = self.net.module.get_lanes(data)
        return lanes

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
            print(f"Saving output to: {out_file}")
        
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        print(f"Number of lanes to draw: {len(lanes)}")
        for i, lane in enumerate(lanes):
            print(f"Lane {i} coordinates: {lane}")
        
        img_with_lanes = data['ori_img'].copy()
        imshow_lanes(img_with_lanes, lanes, show=self.cfg.show)
        h = data['h']
        w = data['w']
        img_with_lanes = cv2.resize(img_with_lanes, (w, h))
        if out_file:
            cv2.imwrite(out_file, img_with_lanes)
            print(f"Image saved at: {out_file}")

    def run(self, img_path):
        data = self.preprocess(img_path)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))
    elif os.path.isdir(p):
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        paths = sorted(
            [f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(img_extensions)]
        )
    elif os.path.isfile(p):
        paths = [p]
    else:
        raise Exception(f'ERROR: {p} does not exist')
    print(f"Found {len(paths)} images in {path}")
    return paths

def process(config, show, output_dir, input_dir, model):
    cfg = Config.fromfile(config)
    cfg.show = show
    cfg.savedir = output_dir
    cfg.load_from = model
    detect = Detect(cfg)
    
    # Tạo thư mục input_dir nếu chưa tồn tại
    os.makedirs(input_dir, exist_ok=True)
    paths = get_img_paths(input_dir)

    if not paths:
        print(f"Warning: No valid image files found in {input_dir}. Skipping inference.")
    else:
        for p in tqdm(paths, desc="Processing images with PyTorch model"):
            detect.run(p)

if __name__ == '__main__':
    os.makedirs('./vis', exist_ok=True)
    config = '/root/lanedet/configs/ufld/resnet18_culane.py'
    show = False
    output_dir = './vis'
    input_dir = './images/'
    model = '/root/lanedet/work_dirs/CULane/20250410_124434_lr_1e-02_b_16/ckpt/999.pth'
    
    process(config, show, output_dir, input_dir, model)