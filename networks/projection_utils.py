import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
import cv2
import math
from os import makedirs
from os.path import join, exists

# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img

# Based on https://github.com/sunset1995/py360convert
class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), bool)
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):

        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi

        lon, lat = np.meshgrid(lon, lat)

        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)

        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)

        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])

        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)

        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):

        bs, ch, h, w = cube_feat.shape
        assert h == self.face_w and w // 6 == self.face_w

        cube_feat = cube_feat.view(bs, ch, 1,  h, w)
        cube_feat = torch.cat(torch.split(cube_feat, self.face_w, dim=-1), dim=2)

        cube_feat = cube_feat.view([bs, ch, 6, self.face_w, self.face_w])
        sample_grid = torch.cat(bs * [self.sample_grid], dim=0)
        equi_feat = F.grid_sample(cube_feat, sample_grid, padding_mode="border", align_corners=True)

        return equi_feat.squeeze(2)
    
# generate patches in a closed-form
# the transformation and equation is referred from http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype = np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz

def equi2pers(erp_img, fov, nrows, patch_size):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)
    
    if nrows==4:    
        num_rows = 4
        num_cols = [3, 6, 6, 3]
        phi_centers = [-67.5, -22.5, 22.5, 67.5]
    if nrows==6:    
        num_rows = 6
        num_cols = [3, 8, 12, 12, 8, 3]
        phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
    if nrows==3:
        num_rows = 3
        num_cols = [3, 4, 3]
        phi_centers = [-60, 0, 60]     
    if nrows==5:
        num_rows = 5
        num_cols = [3, 6, 8, 6, 3]
        phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
            
    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2

            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    center_p = cp.clone()
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)
    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)
  
    grid_tmp = torch.stack([lon, lat], -1)
    xyz = uv2xyz(grid_tmp)
    xyz = xyz.reshape(num_patch, height, width, 3).transpose(0, 3, 1, 2)
    xyz = torch.from_numpy(xyz).to(pers.device).contiguous()
    
    uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
    uv = uv.contiguous()
    return pers, xyz, uv, center_p

def pers2equi(pers_img, fov, nrows, patch_size, erp_size, layer_name):
    bs = pers_img.shape[0]
    channel = pers_img.shape[1]
    device=pers_img.device
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    erp_h, erp_w = pair(erp_size)
    n_patch = pers_img.shape[-1]     
    grid_dir = './grid'
    if not exists(grid_dir):
        makedirs(grid_dir)
    grid_file = join(grid_dir, layer_name + '.pth')  
    
    if not exists(grid_file):  
        FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5
        PI2 = math.pi * 2

        if nrows==4:    
            num_rows = 4
            num_cols = [3, 6, 6, 3]
            phi_centers = [-67.5, -22.5, 22.5, 67.5]
        if nrows==6:    
            num_rows = 6
            num_cols = [3, 8, 12, 12, 8, 3]
            phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
        if nrows==3:
            num_rows = 3
            num_cols = [3, 4, 3]
            phi_centers = [-59.6, 0, 59.6]
        if nrows==5:
            num_rows = 5
            num_cols = [3, 6, 8, 6, 3]
            phi_centers = [-72.2, -36.1, 0, 36.1, 72.2] 
        phi_interval = 180 // num_rows
        all_combos = []

        for i, n_cols in enumerate(num_cols):
            for j in np.arange(n_cols):
                theta_interval = 360 / n_cols
                theta_center = j * theta_interval + theta_interval / 2

                center = [theta_center, phi_centers[i]]
                all_combos.append(center)
                
                
        all_combos = np.vstack(all_combos) 
        n_patch = all_combos.shape[0]
        
        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

        cp = center_point * 2 - 1
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)
        
        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
        lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 
        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
        new_x = new_x / FOV[0] / PI   # -1 to 1
        new_y = new_y / FOV[1] / PI_2
        cos_c_mask = cos_c.reshape(n_patch, erp_h, erp_w)
        cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)
        
        w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32)

        new_x_patch = (new_x + 1) * 0.5 * height
        new_y_patch = (new_y + 1) * 0.5 * width 
        new_x_patch = new_x_patch.reshape(n_patch, erp_h, erp_w)
        new_y_patch = new_y_patch.reshape(n_patch, erp_h, erp_w)
        mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
        mask *= cos_c_mask

        x0 = torch.floor(new_x_patch).type(torch.int64)
        x1 = x0 + 1
        y0 = torch.floor(new_y_patch).type(torch.int64)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width-1)
        x1 = torch.clamp(x1, 0, width-1)
        y0 = torch.clamp(y0, 0, height-1)
        y1 = torch.clamp(y1, 0, height-1)

        wa = (x1.type(torch.float32)-new_x_patch) * (y1.type(torch.float32)-new_y_patch)
        wb = (x1.type(torch.float32)-new_x_patch) * (new_y_patch-y0.type(torch.float32))
        wc = (new_x_patch-x0.type(torch.float32)) * (y1.type(torch.float32)-new_y_patch)
        wd = (new_x_patch-x0.type(torch.float32)) * (new_y_patch-y0.type(torch.float32))

        wa = wa * mask.expand_as(wa)
        wb = wb * mask.expand_as(wb)
        wc = wc * mask.expand_as(wc)
        wd = wd * mask.expand_as(wd)

        w_list[..., 0] = wa
        w_list[..., 1] = wb
        w_list[..., 2] = wc
        w_list[..., 3] = wd

       
        save_file = {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1, 'w_list': w_list, 'mask':mask}
        torch.save(save_file, grid_file)
    else:
        # the online merge really takes time
        # pre-calculate the grid for once and use it during training
        load_file = torch.load(grid_file)
        #print('load_file')
        x0 = load_file['x0']
        y0 = load_file['y0']
        x1 = load_file['x1']
        y1 = load_file['y1']
        w_list = load_file['w_list']
        mask = load_file['mask']

    w_list = w_list.to(device)
    mask = mask.to(device)
    z = torch.arange(n_patch)
    z = z.reshape(n_patch, 1, 1)
    Ia = pers_img[:, :, y0, x0, z]
    Ib = pers_img[:, :, y1, x0, z]
    Ic = pers_img[:, :, y0, x1, z]
    Id = pers_img[:, :, y1, x1, z]
    output_a = Ia * mask.expand_as(Ia)
    output_b = Ib * mask.expand_as(Ib)
    output_c = Ic * mask.expand_as(Ic)
    output_d = Id * mask.expand_as(Id)

    output_a = output_a.permute(0, 1, 3, 4, 2)
    output_b = output_b.permute(0, 1, 3, 4, 2)
    output_c = output_c.permute(0, 1, 3, 4, 2)
    output_d = output_d.permute(0, 1, 3, 4, 2)   
    w_list = w_list.permute(1, 2, 0, 3)
    w_list = w_list.flatten(2)
    w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
    w_list = F.normalize(w_list, p=1, dim=-1).reshape(erp_h, erp_w, n_patch, 4)
    w_list = w_list.unsqueeze(0).unsqueeze(0)
    output = output_a * w_list[..., 0] + output_b * w_list[..., 1] + \
        output_c * w_list[..., 2] + output_d * w_list[..., 3]
    img_erp = output.sum(-1) 

    return img_erp

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp, W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img