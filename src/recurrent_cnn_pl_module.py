import torch
import pytorch_lightning as pl
import torchmetrics.image
import torchmetrics
from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
import os.path as osp
from einops import rearrange

from losses import CharbonnierLoss, PerceptualLoss


class RecurrentCNNModule(pl.LightningModule):

    def __init__(self, opt, generator=None, window_size=5, pixel_loss_weight=200, perceptual_loss_weight=1):
        super(RecurrentCNNModule, self).__init__()
        self.save_hyperparameters(ignore=["generator"])
        self.opt = opt
        self.window_size = window_size

        self.net_g = generator

        self.lr = 1.9e-5
        weight_pixel_criterion = pixel_loss_weight
        self.pixel_criterion = CharbonnierLoss(loss_weight=weight_pixel_criterion)

        vgg_layer_weights = {'conv5_4': 1, 'relu4_4': 1, 'relu3_4': 1, 'relu2_2': 1}
        weight_perceptual_criterion = perceptual_loss_weight
        self.perceptual_criterion = PerceptualLoss(layer_weights=vgg_layer_weights, loss_weight=weight_perceptual_criterion)

        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex")

    def forward(self, *x):
        return self.net_g(*x)

    def training_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        imgs_gt = batch["imgs_gt"]
        outputs_g = self.net_g(imgs_lq)

        B, T, C, H, W = outputs_g.shape
        outputs_g = rearrange(outputs_g, 'b t c h w -> (b t) c h w', b=B, t=T)
        imgs_gt = rearrange(imgs_gt, 'b t c h w -> (b t) c h w', b=B, t=T)

        pixel_loss_g = self.pixel_criterion(outputs_g, imgs_gt)
        perceptual_loss_g = self.perceptual_criterion(outputs_g, imgs_gt)

        total_loss_g = pixel_loss_g + perceptual_loss_g

        log_loss_g = {"total_g": total_loss_g,
                      "pixel_g": pixel_loss_g,
                      "perceptual_g": perceptual_loss_g}
        self.log_dict(log_loss_g, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=imgs_gt.shape[0])
        return total_loss_g

    def validation_step(self, batch, batch_idx):
        imgs_lq = batch["imgs_lq"]
        imgs_gt = batch["imgs_gt"]
        outputs_g = self.net_g(imgs_lq).to(torch.float32)

        psnr, ssim, lpips = 0., 0., 0.
        for i, output_g in enumerate(outputs_g):
            output_g = torch.clamp(output_g, 0, 1)
            img_gt = imgs_gt[i]
            psnr += self.psnr(output_g, img_gt)
            ssim += self.ssim(output_g, img_gt, data_range=1.)
            with torch.no_grad():
                lpips += self.lpips(output_g * 2 - 1, img_gt * 2 - 1)    # Input must be in [-1, 1] range
        psnr /= len(outputs_g)
        ssim /= len(outputs_g)
        lpips /= len(outputs_g)

        log_metrics = {"psnr": psnr,
                       "ssim": ssim,
                       "lpips": lpips}

        self.log_dict(log_metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=imgs_gt.shape[0])

        imgs_name = batch["img_name"]
        if batch_idx == 0:
            imgs_name = [imgs_name[0]]
        for i, img_name in enumerate(imgs_name):
            img_num = int(osp.basename(img_name)[:-4])
            if img_num % 100 == 0 or batch_idx == 0:
                single_img_lq = imgs_lq[0, self.window_size // 2]
                single_img_gt = imgs_gt[0, self.window_size // 2]
                single_img_output = torch.clamp(outputs_g[0, self.window_size // 2], 0., 1.)
                concatenated_img = torch.cat((single_img_lq, single_img_output, single_img_gt), -1)
                self.logger.experiment.log_image(to_pil_image(concatenated_img.cpu()), str(img_num), step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.psnr.reset()
        self.lpips.reset()

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(self.net_g.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.99))
        return optimizer_g
