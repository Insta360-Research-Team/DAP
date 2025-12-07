from __future__ import absolute_import, division, print_function
import os
import cv2
import sys 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(PROJECT_ROOT)
import sys
import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import torch.nn as nn
from networks.models import *
import OpenEXR
import Imath

# --------------------- 工具函数 ---------------------

# def read_exr_depth(exr_path):
#     exr_file = OpenEXR.InputFile(exr_path)
#     header = exr_file.header()
#     dw = header['dataWindow']
#     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
#     FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
#     depth_str = exr_file.channel('R', FLOAT)  
#     depth = np.frombuffer(depth_str, dtype=np.float32)
#     depth = np.reshape(depth, (size[1], size[0]))
    
#     return depth

def colorize_depth(depth, colormap=cv2.COLORMAP_JET):
    depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
    depth_color = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_color, colormap)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_model(config):
    """加载模型"""
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    print(f"🔹 Loading model weights from: {model_path}")
    model_dict = torch.load(model_path, map_location='cuda')

    model = make(config['model'])
    if any(k.startswith('module') for k in model_dict.keys()):
        model = nn.DataParallel(model)
    model.cuda()
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()
    print("✅ Model loaded successfully.\n")
    return model


def infer_and_save(model, img_path, out_root, idx):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Cannot read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, dict) and 'pred_depth' in outputs:
            pred = outputs['pred_depth'][0].detach().cpu().squeeze().numpy()
        else:
            pred = outputs[0].detach().cpu().squeeze().numpy()

    filename = f"{idx:06d}"  

    pred_npy_path = os.path.join(out_root, "depth_npy", filename + ".npy")
    pred_png_path = os.path.join(out_root, "depth_vis", filename + ".png")

    ensure_dir(pred_npy_path)
    ensure_dir(pred_png_path)

    np.save(pred_npy_path, pred)
    
    pred_vis = torch.clamp(torch.from_numpy(pred), min=0.001, max=1.0)
    print("pred_vis.min(), pred_vis.max()", pred_vis.min(), pred_vis.max())
    depth_vis = (pred_vis.squeeze().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(pred_png_path, depth_vis)
    

    
    print(f"✅ Saved pred only: {filename}")


# --------------------- 主函数 ---------------------

def main(config_path, txt_path, out_root):
    # 读取配置
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("✅ Config loaded.")

    # 加载模型
    model = load_model(config)

    # 读取txt
    with open(txt_path, 'r') as f:
        img_list = [l.strip() for l in f.readlines() if l.strip()]

    # 将相对路径转换为绝对路径（相对于数据集根目录）
    # img_list = [os.path.join(data_root, p) if not os.path.isabs(p) else p for p in img_list]

    # print(f"🔹 Total images to infer: {len(img_list)}\n")

    for idx, img_path in enumerate(tqdm(img_list, desc="Inferencing"), start=1):
        infer_and_save(model, img_path, out_root, idx)

    print(f"\n🎯 推理完成！")
    print(f"   预测深度保存在: {out_root}/depth_npy 和 {out_root}/depth_vis")
    print(f"   GT深度保存在: {out_root}/gt_depth_npy 和 {out_root}/gt_depth_vis")

# --------------------- 程序入口 ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/infer.yaml')
    parser.add_argument('--txt', default='datasets/dit_haoran.txt')
    parser.add_argument('--output', default='just_try')
    parser.add_argument('--gpu', default='0', help='使用的GPU编号')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args.config, args.txt, args.output)
