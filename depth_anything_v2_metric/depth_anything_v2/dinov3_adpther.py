# dinov3_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3Adapter(nn.Module):
    """
    DINOv3 Adapter:
    """

    MODEL_MAP = {
        "vits": "dinov3_vits16",
        "vitb": "dinov3_vitb16",
        "vitl": "dinov3_vitl16",
        "vitg": "dinov3_vitg14",   
        "vit7b": "dinov3_vit7b16",
    }

    def __init__(self, model_name, repo_dir, arch=None, weight_path=None):
        super().__init__()

        if arch is None:
            if model_name not in self.MODEL_MAP:
                raise ValueError(f"Unknown model_name={model_name}, must be one of {list(self.MODEL_MAP.keys())}")
            arch = self.MODEL_MAP[model_name]

        self.model = torch.hub.load(repo_dir, arch, source="local", pretrained=False)
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            raise AttributeError("DINOv3 model missing embed_dim")

        self.patch_size = getattr(self.model, "patch_size", None)
        if self.patch_size is None:
            pe = getattr(self.model, "patch_embed", None)
            if pe is not None and hasattr(pe, "patch_size"):
                ps = pe.patch_size
                self.patch_size = ps if isinstance(ps, int) else ps[0]
        if self.patch_size is None:
            raise AttributeError("DINOv3 model missing patch_size")
        
        self.blocks = getattr(self.model, "blocks", None)
        if self.blocks is None:
            raise AttributeError("DINOv3 model missing blocks")

        self.n_blocks = getattr(self.model, "n_blocks", len(self.blocks))
        self.depth = self.n_blocks

        self.norm = nn.LayerNorm(self.embed_dim)

    # @torch.no_grad()
    def get_intermediate_layers(
        self, x, n=1, return_class_token=False, norm=True
    ):
        outputs = self.model.get_intermediate_layers(
            x, n=n, reshape=False, return_class_token=True, norm=norm
        )

        patch_maps, cls_tokens = [], []
        H, W = x.shape[-2], x.shape[-1]
        h, w = H // self.patch_size, W // self.patch_size  

        for (out_all, out_cls) in outputs:
            if norm:
                out_all = self.norm(out_all)

            out_patches = out_all[:, 1:, :]   # [B, N, C]
            B, N, C = out_patches.shape
            sqrtN = int(N ** 0.5)
            if sqrtN * sqrtN == N:
                grid = out_patches.transpose(1, 2).reshape(B, C, sqrtN, sqrtN)
            else:
                grid = out_patches.transpose(1, 2).reshape(B, C, N, 1)
                grid = F.interpolate(grid, size=(h * w, 1), mode="bilinear").squeeze(-1)
                grid = grid.reshape(B, C, h, w)

            if grid.shape[-2:] != (h, w):
                grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)

            patch_maps.append(grid.contiguous())
            cls_tokens.append(out_cls)

        if return_class_token:
            return tuple(zip(patch_maps, cls_tokens))
        return tuple(patch_maps)
