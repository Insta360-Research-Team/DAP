import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2_metric.depth_anything_v2.dinov3_adpther import DINOv3Adapter
from argparse import Namespace
from .models import register
from depth_anything_utils import Resize, NormalizeImage, PrepareForNet

class DAP(nn.Module):
    def __init__(self, args):
        super().__init__()
        midas_model_type = args.midas_model_type
        fine_tune_type = args.fine_tune_type
        min_depth = args.min_depth
        self.max_depth = args.max_depth
        train_decoder = args.train_decoder

        # Pre-defined setting of the model
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Load the pretrained model of depth anything
        dinov3_repo_dir="/home/tione/notebook/home/wenxuan/DAM_daual_clean/depth_anything_v2_metric/depth_anything_v2/dinov3"     # 你的本地 repo
        dinov3_arch="dinov3_vitl16"          
        dinov3_weight="/home/tione/notebook/home/gavinqi/wenxuan/DINOv3/DINOv3 ViT LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

        depth_anything = DepthAnythingV2(
            **{**model_configs[midas_model_type], 'max_depth': 1.0},
            dinov3_repo_dir=dinov3_repo_dir,
            dinov3_arch=dinov3_arch,
            dinov3_weight=dinov3_weight
        )


        self.core = depth_anything
        for param in self.core.parameters():
            param.requires_grad = True


    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        erp_pred, mask_pred = self.core(image)
        erp_pred = erp_pred.unsqueeze(1)
        erp_pred[erp_pred < 0] = 0
        mask_pred = mask_pred.unsqueeze(1)
        outputs = {}
        outputs["pred_depth"] = erp_pred * self.max_depth
        outputs["pred_mask"] = mask_pred


        return outputs

    def get_encoder_decoder_params(self):
        encoder_params = list(self.core.pretrained.parameters())
        decoder_params = list(self.core.depth_head.parameters())
        mask_params = list(self.core.mask_head.parameters())

        return encoder_params, decoder_params, mask_params

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)["pred_depth"]
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size * 2,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=self.core.patch_size,
                # ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    
@register('dap')
def make_model(midas_model_type='vitl', fine_tune_type='none', min_depth=0.001, max_depth=1.0, train_decoder=True):
    args = Namespace()
    args.midas_model_type = midas_model_type
    args.fine_tune_type = fine_tune_type
    args.min_depth = min_depth
    args.max_depth = max_depth
    args.train_decoder = train_decoder
    return DAP(args)