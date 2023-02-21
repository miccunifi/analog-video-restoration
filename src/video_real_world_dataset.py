import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from os import path as osp
from glob import glob
from PIL import Image
from torchvision.transforms import ToTensor
from pathlib import Path

from utils import consistent_crop


class VideoRealWorldDataset(Dataset):
    """
    Video dataset where each item is given by window_size LQ frames

    :param dataroot_lq (str): data root path for lq frames
    :param dataroot_gt (str): data root path for gt frames
    :param window_size (int): window size for input frames (even numbers are transformed to the following odd number)
                             Default: 5
    :param gt_patch_size (int): size of gt crops. Default: 256
    :param crop_mode (str): crop modality. Supported modes: random, center
    :param scale (int): scale of gt crops w.r.t. lq crops. Default: 1

    :return dict["imgs_lq"]: LQ center frame + window_size//2 left and right neighboring frames (as Torch tensors)
    :return dict["img_name"]: Center frame name
    """
    def __init__(self, dataroot_lq, window_size=5, downscale=1, patch_size=256, crop_mode="center"):
        self.window_size = window_size
        self.downscale = downscale
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.clip_name = Path(dataroot_lq).name
        self.img_paths_lq = sorted(glob(osp.join(dataroot_lq, "*.jpg")), key=lambda x: int(osp.basename(x)[:-4]))

    def __getitem__(self, idx):
        center_frame_lq = self.img_paths_lq[idx]
        img_name = osp.basename(center_frame_lq)
        half_window_size = self.window_size // 2

        idxs_imgs_lq = np.arange(idx - half_window_size, idx + half_window_size + 1)
        idxs_imgs_lq = list(idxs_imgs_lq[(idxs_imgs_lq >= 0) & (idxs_imgs_lq < len(self.img_paths_lq))])
        imgs_lq = []
        for img_idx in idxs_imgs_lq:
            img_t = ToTensor()(Image.open(self.img_paths_lq[img_idx]))
            img_t = F.interpolate(img_t.unsqueeze(0), scale_factor=1/self.downscale, mode="bilinear").squeeze(0)
            imgs_lq.append(img_t)

        if len(imgs_lq) < self.window_size:
            black_frame = torch.zeros_like(imgs_lq[0])
            missing_frames_left = half_window_size - idx
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
            missing_frames_right = half_window_size - (len(self.img_paths_lq) - idx)
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)

        imgs_lq = consistent_crop([imgs_lq], patch_size=self.patch_size, crop_mode=self.crop_mode)
        # img_lqs: (c, self.window_size, h, w)
        # img_name: (str)
        return {"imgs_lq": imgs_lq, "img_name": f"{self.clip_name}/{img_name}"}

    def __len__(self):
        return len(self.img_paths_lq)
