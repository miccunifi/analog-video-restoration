import numpy
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import os
import os.path as osp
import cv2
from argparse import ArgumentParser

from video_real_world_dataset import VideoRealWorldDataset
from video_swin_unet import VideoSwinEncoderDecoder


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):

    checkpoints_path = Path("../pretrained_models") / args.experiment_name
    checkpoint_file = os.listdir(checkpoints_path)[-1]
    checkpoint = checkpoints_path / checkpoint_file

    results_path = Path(args.results_path) / Path(f"{args.experiment_name}")
    results_path.mkdir(parents=True, exist_ok=True)
    video_path = results_path / Path("videos")
    restored_video_path = video_path / "restored"
    restored_video_path.mkdir(parents=True, exist_ok=True)
    combined_video_path = video_path / "combined"
    combined_video_path.mkdir(parents=True, exist_ok=True)

    test_dataset = VideoRealWorldDataset(args.data_base_path,
                                         window_size=5,
                                         downscale=1,
                                         patch_size=args.patch_size,
                                         crop_mode="center")

    model = VideoSwinEncoderDecoder(use_checkpoint=args.use_checkpoint, depths=[2, 2, 6, 2], embed_dim=96)
    state_dict = torch.load(checkpoint)["state_dict"]
    state_dict = dict([(k[len("net_g."):], v) for k, v in state_dict.items() if k.startswith("net_g.")])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)

    last_clip = ""
    restored_video_writer = cv2.VideoWriter()
    combined_video_writer = cv2.VideoWriter()

    for batch in dataloader:
        imgs_lq = batch["imgs_lq"]
        h, w = imgs_lq.shape[-2:]
        pad_left_right = (w - (
                    w // 2 ** 4) * 2 ** 4) // 2  # Input frames must be divisible by 16 (due to the 4 downsampling operations)
        pad_top_bottom = (h - (
                    h // 2 ** 4) * 2 ** 4) // 2  # Input frames must be divisible by 16 (due to the 4 downsampling operations)
        pad = (pad_left_right, pad_left_right, pad_top_bottom, pad_top_bottom)
        imgs_lq = F.pad(imgs_lq, pad=pad, mode="constant", value=0)
        imgs_lq = imgs_lq.to(device)
        input_size = imgs_lq.shape
        img_name = batch["img_name"]
        print("Restoring: ", img_name[0])

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = torch.clamp(model(imgs_lq), 0, 1)
                output = output[:, 5 // 2].permute(0, 2, 3, 1).cpu().numpy()

        for i in range(output.shape[0]):

            video_clip = osp.dirname(img_name[i])
            (results_path / Path(video_clip)).mkdir(parents=True, exist_ok=True)

            restored = (output[i] * 255).astype(np.uint8)
            restored = restored[..., ::-1]  # RGB -> BGR
            restored = restored[pad_top_bottom:(h - pad_top_bottom), pad_left_right:(w - pad_left_right), :]

            input = imgs_lq[i, 5 // 2].permute(1, 2, 0).cpu().numpy()
            input = (input * 255).astype(np.uint8)
            input = input[..., ::-1]  # RGB -> BGR
            input = input[pad_top_bottom:(h - pad_top_bottom), pad_left_right:(w - pad_left_right), :]

            if video_clip != last_clip:
                restored_video_writer.release()
                combined_video_writer.release()
                last_clip = video_clip
                restored_video_writer = cv2.VideoWriter(f"{restored_video_path}/{video_clip}.mp4",
                                                        cv2.VideoWriter_fourcc(*'mp4v'), 25,
                                                        restored.shape[0:2])
                combined_shape = (restored.shape[0] * 2, restored.shape[1])
                combined_video_writer = cv2.VideoWriter(f"{combined_video_path}/{video_clip}.mp4",
                                                        cv2.VideoWriter_fourcc(*'mp4v'), 25,
                                                        combined_shape)

            restored_video_writer.write(restored)
            combined = numpy.hstack((input, restored))
            combined_video_writer.write(combined)
            cv2.imwrite(f"{results_path}/{img_name[i]}", restored)

    restored_video_writer.release()
    combined_video_writer.release()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="video_swin_unet")
    parser.add_argument("--data-base-path", type=str)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--no-checkpoint", default=True, action="store_false")
    args = parser.parse_args()

    main(args)
