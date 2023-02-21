import numpy as np
import torch
from torch.utils.data import Dataset
from os import path as osp
from glob import glob
from PIL import Image
from torchvision.transforms import ToTensor
import lmdb

from utils import paired_crop


class VideoRecurrentDataset(Dataset):
    """
    Video recurrent dataset where each item is given by window_size LQ and GT frames

    :param dataroot_lq (str): data root path for lq frames
    :param dataroot_gt (str): data root path for gt frames
    :param window_size (int): window size for input frames (even numbers are transformed to the following odd number).
                            Default: 5
    :param gt_patch_size (int): size of gt crops. Default: 256
    :param crop_mode (str): crop modality. Supported modes: random, center
    :param scale (int): scale of gt crops w.r.t. lq crops. Default: 1

    :return dict["imgs_lq"]: LQ center frame + window_size//2 left and right neighboring frames (as Torch tensors)
    :return dict["imgs_gt"]: GT center frame + window_size//2 left and right neighboring frames (as Torch tensors)
    """
    def __init__(self, dataroot_lq, dataroot_gt, window_size=5, frame_offset=1, gt_patch_size=256, crop_mode="random", use_lmdb=False, scale=1):
        self.root_lq = dataroot_lq
        self.root_gt = dataroot_gt
        self.window_size = window_size
        self.frame_offset = frame_offset
        self.gt_patch_size = gt_patch_size
        self.crop_mode = crop_mode
        self.scale = scale
        self.img_paths_lq = []
        self.img_paths_gt = []
        self.clip_intervals = {}

        subfolders_lq = sorted(glob(osp.join(self.root_lq, '*')))
        subfolders_gt = sorted(glob(osp.join(self.root_gt, '*')))
        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            img_paths_lq = sorted(glob(osp.join(subfolder_lq, "*.jpg")), key=lambda x: int(osp.basename(x)[:-4]))
            img_paths_gt = sorted(glob(osp.join(subfolder_gt, "*.jpg")), key=lambda x: int(osp.basename(x)[:-4]))
            assert len(img_paths_lq) == len(img_paths_gt), f"{subfolder_lq} and {subfolder_gt} contain different number of images"
            if len(img_paths_lq) == 0:
                continue
            clip_interval_start = len(self.img_paths_lq)
            self.img_paths_lq.extend(img_paths_lq)
            clip_interval_end = len(self.img_paths_lq) - 1
            self.img_paths_gt.extend(img_paths_gt)
            clip_name = osp.basename(subfolder_lq)
            self.clip_intervals[clip_name] = (clip_interval_start, clip_interval_end)

    def __getitem__(self, idx):
        center_frame_lq = self.img_paths_lq[idx]
        img_name = osp.basename(center_frame_lq)
        clip_name = osp.basename(osp.dirname(center_frame_lq))
        clip_interval_start, clip_interval_end = self.clip_intervals[clip_name]
        half_window_size = self.window_size // 2

        idxs_imgs_lq = np.arange(idx - half_window_size*self.frame_offset, idx + (half_window_size*self.frame_offset) + 1, self.frame_offset)
        idxs_imgs_lq = list(idxs_imgs_lq[(idxs_imgs_lq >= clip_interval_start) & (idxs_imgs_lq <= clip_interval_end)])
        imgs_lq = []
        imgs_gt = []

        for img_idx in idxs_imgs_lq:
            img_t = ToTensor()(Image.open(self.img_paths_lq[img_idx]))
            imgs_lq.append(img_t)
            img_t = ToTensor()(Image.open(self.img_paths_gt[img_idx]))
            imgs_gt.append(img_t)

        if len(imgs_lq) < self.window_size:
            black_frame = torch.zeros_like(imgs_lq[0])
            # missing_frames_left = half_window_size - (idx - clip_interval_start)
            missing_frames_left = half_window_size - idxs_imgs_lq.index(idx)
            for _ in range(missing_frames_left):
                imgs_lq.insert(0, black_frame)
                imgs_gt.insert(0, black_frame)
            # missing_frames_right = half_window_size - (clip_interval_end - idx)
            missing_frames_right = half_window_size - (len(idxs_imgs_lq) - 1 - idxs_imgs_lq.index(idx))
            for _ in range(missing_frames_right):
                imgs_lq.append(black_frame)
                imgs_gt.append(black_frame)
        imgs_lq = torch.stack(imgs_lq)
        imgs_gt = torch.stack(imgs_gt)

        if self.crop_mode == "random":
            imgs_gt, imgs_lq = paired_crop(imgs_gt, imgs_lq, gt_patch_size=self.gt_patch_size, crop_mode=self.crop_mode,
                                          scale=self.scale)
        elif self.crop_mode == "center":
            imgs_gt, imgs_lq = paired_crop(imgs_gt, imgs_lq, gt_patch_size=self.gt_patch_size, crop_mode=self.crop_mode,
                                          scale=self.scale)

        # img_lqs: (c, self.window_size, h, w)
        # imgs_gt: (c, self.windows_size, h, w)
        # img_name: (str)
        return {"imgs_lq": imgs_lq, "imgs_gt": imgs_gt, "img_name": f"{clip_name}/{img_name}"}

    def __len__(self):
        return len(self.img_paths_lq)
