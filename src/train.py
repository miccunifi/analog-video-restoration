import comet_ml
import os
import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from argparse import ArgumentParser

from video_swin_unet import VideoSwinEncoderDecoder
from video_recurrent_dataset import VideoRecurrentDataset
from video_data_pl_module import VideoDataModule
from recurrent_cnn_pl_module import RecurrentCNNModule
from utils import init_logger


def main(args):
    training_params = args.training_params
    data_params = args.data_params
    model_params = args.model_params

    pl.seed_everything(42, workers=True)
    os.environ['PYTHONHASHSEED'] = str(42)

    experiment_name = args.experiment_name
    experiment_key = args.experiment_key
    online_logger = not args.offline

    data_base_path = Path(args.data_base_path)

    training_input_path = data_base_path / "train" / "input"
    training_gt_path = data_base_path / "train" / "gt"
    val_input_path = data_base_path / "val" / "input"
    val_gt_path = data_base_path / "val" / "gt"

    checkpoints_path = Path(args.checkpoints_path) / experiment_name
    logger = init_logger(api_key=args.api_key, experiment_name=experiment_name, experiment_key=experiment_key, online=online_logger)
    args.training_params["logger"] = logger
    logger.experiment.log_parameters(training_params, prefix="training")
    logger.experiment.log_parameters(data_params, prefix="data")

    train_dataset = VideoRecurrentDataset(training_input_path, training_gt_path,
                                          window_size=data_params["window_size"],
                                          frame_offset=data_params["frame_offset"],
                                          gt_patch_size=data_params["gt_patch_size"],
                                          crop_mode="random")
    val_dataset = VideoRecurrentDataset(val_input_path, val_gt_path,
                                         window_size=data_params["window_size"],
                                         frame_offset=data_params["frame_offset"],
                                         gt_patch_size=data_params["gt_patch_size"],
                                         crop_mode="center")
    data_module = VideoDataModule(train_dataset, val_dataset, batch_size=data_params["batch_size"],
                                  num_workers=data_params["num_workers"])

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path,
                                          filename="{epoch}-{step}-{lpips:.3f}",
                                          save_weights_only=False,
                                          monitor="lpips",
                                          save_top_k=1,
                                          save_last=True)

    generator = VideoSwinEncoderDecoder(use_checkpoint=True, depths=[2, 2, 6, 2], embed_dim=96)
    model = RecurrentCNNModule(opt=None, generator=generator,
                               window_size=data_params["window_size"],
                               pixel_loss_weight=model_params["pixel_loss_weight"],
                               perceptual_loss_weight=model_params["perceptual_loss_weight"])

    trainer = Trainer(**training_params, callbacks=[checkpoint_callback])

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="video_swin_unet")
    parser.add_argument("--data-base-path", type=str)
    parser.add_argument("--checkpoints-path", type=str, default="pretrained_models")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--resume-from-checkpoint", default=False, action="store_true")
    parser.add_argument("--resume-checkpoint-filename", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--offline", default=False, action="store_true")
    parser.add_argument("--experiment-key", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--pixel-loss-weight", type=float, default=200)
    parser.add_argument("--perceptual-loss-weight", type=float, default=1)
    parser.add_argument("--no-ddp-strategy", default=False, action="store_true")
    args = parser.parse_args()

    training_params = {
        "benchmark": True,
        "precision": 16,
        "log_every_n_steps": 50,
        "accelerator": "gpu",
        "devices": args.devices,
        "strategy": None if args.no_ddp_strategy else DDPStrategy(find_unused_parameters=True, static_graph=True),
        "max_epochs": args.num_epochs,
        "resume_from_checkpoint": None if not args.resume_from_checkpoint else Path(args.checkpoints_path) / args.experiment_name / args.resume_checkpoint_filename
    }

    data_params = {
        "window_size": 5,
        "frame_offset": 1,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "gt_patch_size": 128
    }

    model_params = {
        "pixel_loss_weight": args.pixel_loss_weight,
        "perceptual_loss_weight": args.perceptual_loss_weight,
    }

    args.training_params = training_params
    args.data_params = data_params
    args.model_params = model_params

    main(args)

