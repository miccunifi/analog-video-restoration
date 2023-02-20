import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from os import path as osp
from glob import glob
from PIL import Image
from torchvision.transforms import ToTensor
import random


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
        self.root_lq = dataroot_lq
        self.window_size = window_size
        self.downscale = downscale
        self.patch_size = patch_size
        self.crop_mode = crop_mode

        self.img_paths_lq = []
        self.img_paths_gt = []
        self.clip_intervals = {}

        subfolders_lq = sorted(glob(osp.join(self.root_lq, '*')))
        for subfolder_lq in subfolders_lq:
            img_paths_lq = sorted(glob(osp.join(subfolder_lq, "*.jpg")), key=lambda x: int(osp.basename(x)[:-4]))
            if len(img_paths_lq) == 0:
                continue
            clip_interval_start = len(self.img_paths_lq)
            self.img_paths_lq.extend(img_paths_lq)
            clip_interval_end = len(self.img_paths_lq) - 1
            clip_name = osp.basename(subfolder_lq)
            self.clip_intervals[clip_name] = (clip_interval_start, clip_interval_end)

    def __getitem__(self, idx):
        center_frame_lq = self.img_paths_lq[idx]
        img_name = osp.basename(center_frame_lq)
        clip_name = osp.basename(osp.dirname(center_frame_lq))
        clip_interval_start, clip_interval_end = self.clip_intervals[clip_name]
        half_window_size = self.window_size // 2

        idxs_imgs_lq = np.arange(idx - half_window_size, idx + half_window_size + 1)
        idxs_imgs_lq = list(idxs_imgs_lq[(idxs_imgs_lq >= clip_interval_start) & (idxs_imgs_lq <= clip_interval_end)])
        imgs_lq = []
        for img_idx in idxs_imgs_lq:
            img_t = ToTensor()(Image.open(self.img_paths_lq[img_idx]))
            img_t = F.interpolate(img_t.unsqueeze(0), scale_factor=1/self.downscale, mode="bilinear").squeeze(0)
            imgs_lq.append(img_t)

        if len(imgs_lq) < self.window_size:
            black_frame = torch.zeros_like(imgs_lq[0])
            missing_frames_left = half_window_size - (idx - clip_interval_start)
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
            missing_frames_right = half_window_size - (clip_interval_end - idx)
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)

        imgs_lq = consistent_crop([imgs_lq], patch_size=self.patch_size, crop_mode=self.crop_mode)
        # img_lqs: (c, self.window_size, h, w)
        # img_name: (str)
        return {"imgs_lq": imgs_lq, "img_name": f"{clip_name}/{img_name}"}

    def __len__(self):
        return len(self.img_paths_lq)


def consistent_crop(imgs_list, patch_size=256, crop_mode="random"):
    """Consisten crop. Support Numpy array and Tensor inputs.

    Taken from https://github.com/xinntao/BasicSR/blob/6697f41600769d43ea201db5bc02100c095d682f/basicsr/data/transforms.py
    Args:
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is a ndarray, it will
            be transformed to a list containing itself.
        patch_size (int): patch size.
        crop_mode (str): crop modality. Supported modes: random, center.

    Returns:
        list[ndarray] | ndarray: LQ images. If returned results
            only have one element, just return ndarray.
    """
    if not isinstance(imgs_list, list):
        imgs_list = [imgs_list]
    imgs_list = [[imgs] for imgs in imgs_list]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs_list[0][0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = imgs_list[0][0].size()[-2:]
    else:
        h_lq, w_lq = imgs_list[0][0].shape[0:2]

    if h_lq < patch_size or w_lq < patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({patch_size}, {patch_size}). ')

    if crop_mode == "random":
        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - patch_size)
        left = random.randint(0, w_lq - patch_size)
    elif crop_mode == "center":
        top = int(round(h_lq - patch_size) / 2.0)
        left = int(round(w_lq - patch_size) / 2.0)
    else:
        raise NotImplementedError(f"Only random and center crop are supported.")

    # crop lq patch
    if input_type == 'Tensor':
        imgs_list = [[v[..., top:top + patch_size, left:left + patch_size] for v in imgs] for imgs in imgs_list]
    else:
        imgs_list = [[v[top:top + patch_size, left:left + patch_size, ...] for v in imgs] for imgs in imgs_list]

    imgs_list = [imgs[0] for imgs in imgs_list]
    if len(imgs_list) == 1:
        imgs_list = imgs_list[0]
    return imgs_list
