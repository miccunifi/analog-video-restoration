import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import warnings
import math
from functools import reduce
from operator import mul
import random
from pytorch_lightning.loggers import CometLogger

def init_logger(api_key="", experiment_name="video_swin_unet", experiment_key=None, online=True):
    if online:
        comet_logger = CometLogger(api_key=api_key,
                                   project_name="analog_video_restoration",
                                   experiment_name=experiment_name,
                                   experiment_key=experiment_key)
    else:
        comet_logger = CometLogger(save_dir="comet",
                                    project_name="analog_video_restoration",
                                   experiment_name=experiment_name,
                                   experiment_key=experiment_key)
    return comet_logger


def paired_crop(img_gts, img_lqs, gt_patch_size=256, crop_mode="random", scale=1):
    """Paired crop. Support Numpy array and Tensor inputs.
    It crops lists of lq and gt images with corresponding locations.

    Taken from https://github.com/xinntao/BasicSR/blob/6697f41600769d43ea201db5bc02100c095d682f/basicsr/data/transforms.py
    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        crop_mode (str): crop modality. Supported modes: random, center.
        scale (int): Scale factor.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    if crop_mode == "random":
        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
    elif crop_mode == "center":
        top = int(round(h_lq - gt_patch_size) / 2.0)
        left = int(round(w_lq - gt_patch_size) / 2.0)
    else:
        raise NotImplementedError(f"Only random and center crop are supported.")

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[..., top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[..., top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


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


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.
    From https://github.com/JingyunLiang/VRT/blob/main/models/network_vrt.py
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.
    From https://github.com/JingyunLiang/VRT/blob/main/models/network_vrt.py
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
         w = torch.empty(3, 5)
         nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
