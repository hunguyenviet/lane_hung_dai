import numpy as np
import torch
import cv2
import os
import os.path as osp
import onnxruntime as ort
from lanedet.datasets.process import Process
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.models.registry import build_net
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane

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
        print(f"Config details: show={self.cfg.show}, savedir={self.cfg.savedir}, cut_height={self.cfg.cut_height}")
        # In thêm các tham số có thể liên quan đến vẽ
        print(f"Config visualization params: {vars(self.cfg).get('visualization', 'Not specified')}")

    def preprocess(self, img_path):
        print(f"\n=== Preprocessing image: {img_path} ===")
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            raise FileNotFoundError(f"Could not load image from {img_path}. Please check the file path.")
        h, w = ori_img.shape[0], ori_img.shape[1]
        print(f"Original image size: (width={w}, height={h})")
        
        ori_img = cv2.resize(ori_img, (1640, 590))
        print(f"Image size after resize: (width=1640, height=590)")
        
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        print(f"Image size after cutting (cut_height={self.cfg.cut_height}): {img.shape}")
        
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        print(f"Input shape for PyTorch after processing: {data['img'].shape}")
        
        data.update({'img_path': img_path, 'ori_img': ori_img, 'h': h, 'w': w})
        print(f"Data keys after preprocessing: {list(data.keys())}")
        return data

    def inference_pth(self, data):
        print("\n=== Running PyTorch inference ===")
        with torch.no_grad():
            data = self.net(data)
            lanes = self.net.module.get_lanes(data)
        print(f"Number of lanes detected (PyTorch): {len(lanes)}")
        if lanes and isinstance(lanes[0], list):
            lanes = lanes[0]  # Làm phẳng nếu lồng
            print(f"Number of lanes after flattening (PyTorch): {len(lanes)}")
            for i, lane in enumerate(lanes):
                print(f"Raw PyTorch Lane {i} (before to_array): {lane}")
        return lanes

    def inference_onnx(self, data, session):
        print("\n=== Running ONNX inference ===")
        img_numpy = data['img'].cpu().numpy()
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        print(f"ONNX model inputs: {input_name}")
        print(f"ONNX model outputs: {output_names}")
        outputs = session.run(output_names, {input_name: img_numpy})
        cls_output_tensor = torch.from_numpy(outputs[0]).cuda()
        data_onnx = {'img': data['img'], 'cls': cls_output_tensor}
        lanes = self.net.module.get_lanes(data_onnx)
        print(f"Number of lanes detected (ONNX): {len(lanes)}")
        if lanes and isinstance(lanes[0], list):
            lanes = lanes[0]  # Làm phẳng nếu lồng
            print(f"Number of lanes after flattening (ONNX): {len(lanes)}")
            for i, lane in enumerate(lanes):
                print(f"Raw ONNX Lane {i} (before to_array): {lane}")
        return lanes

    def show(self, data, model_type="PyTorch"):
        print(f"\n=== Showing and saving results ({model_type}) ===")
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']).replace('.jpg', f'_{model_type.lower()}.jpg'))
            print(f"Saving output to: {out_file}")
        
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        print(f"Number of lanes to draw: {len(lanes)}")
        for i, lane in enumerate(lanes):
            print(f"Lane {i} coordinates (after to_array, {model_type}): {lane}")
            print(f"Number of points in Lane {i}: {len(lane)}")
        
        img_with_lanes = data['ori_img'].copy()
        print(f"Image size before drawing: {img_with_lanes.shape}")
        # In các tham số được truyền vào imshow_lanes
        print(f"Parameters passed to imshow_lanes: show={self.cfg.show}")
        # In toàn bộ cfg để kiểm tra các tham số có thể ảnh hưởng đến vẽ
        print(f"Full config (self.cfg): {vars(self.cfg)}")
        imshow_lanes(img_with_lanes, lanes, show=self.cfg.show)
        print(f"Drawing completed with {len(lanes)} lanes")
        
        h = data['h']
        w = data['w']
        img_with_lanes = cv2.resize(img_with_lanes, (w, h))
        print(f"Image size after resizing to original: (width={w}, height={h})")
        
        # In các điểm được vẽ lên ảnh sau cùng
        print(f"\n=== Points drawn on the final {model_type} image ===")
        for i, lane in enumerate(lanes):
            print(f"{model_type} Lane {i} points (final coordinates):")
            for j, point in enumerate(lane):
                print(f"  Point {j}: ({point[0]:.6f}, {point[1]:.6f})")
        
        if out_file:
            cv2.imwrite(out_file, img_with_lanes)
            print(f"Image saved at: {out_file}")

    def run(self, img_path, session):
        print(f"\n=== Starting lane detection for image: {img_path} ===")
        data = self.preprocess(img_path)
        
        # Suy luận PyTorch
        data['lanes'] = self.inference_pth(data)
        if self.cfg.show or self.cfg.savedir:
            self.show(data, model_type="PyTorch")
        
        # Suy luận ONNX
        data['lanes'] = self.inference_onnx(data, session)
        if self.cfg.show or self.cfg.savedir:
            self.show(data, model_type="ONNX")
        
        print(f"=== Finished processing image: {img_path} ===\n")
        return data

def process(config, show, output_dir, image_path, model, onnx_model_path):
    print(f"\n=== Starting process with config: {config} ===")
    print(f"Show: {show}, Output dir: {output_dir}, Image path: {image_path}, Model: {model}, ONNX Model: {onnx_model_path}")
    cfg = Config.fromfile(config)
    cfg.show = show
    cfg.savedir = output_dir
    cfg.load_from = model
    
    # Khởi tạo phiên ONNX
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    print(f"Loaded ONNX model from: {onnx_model_path}")
    
    detect = Detect(cfg)
    detect.run(image_path, session)
    print("=== Process completed ===")

if __name__ == '__main__':
    print("\n=== Main execution started ===")
    os.makedirs('./vis', exist_ok=True)
    config = '/root/lanedet/configs/ufld/resnet18_culane.py'
    show = False
    output_dir = './vis'
    image_path = '/root/train_lanedet/vis/00000.jpg'
    model = '/root/lanedet/ufld_r18_culane.pth'
    onnx_model_path = '/root/lanedet/ufld_r18_culane_fp16.onnx'
    
    process(config, show, output_dir, image_path, model, onnx_model_path)
    print("=== Main execution finished ===")