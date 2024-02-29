import torch
import numpy as np
import cv2
import torch.nn as nn
import argparse
from typing import Any, Dict, List
import sys
from pathlib import Path
from PIL import Image
import os
import tempfile
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from ostrack import build_ostrack_model, get_box_using_ostrack
from sttn_video_inpaint import build_sttn_model, \
    inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask, show_mask, show_points


def setup_args(parser):
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--tracker_ckpt", type=str, required=True,
        help="The path to tracker checkpoint.",
    )
    parser.add_argument(
        "--vi_ckpt", type=str, required=True,
        help="The path to video inpainter checkpoint.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=2, required=True,
        help="Which mask in the first frame to determine the inpaint region.",
    )
    parser.add_argument(
        "--fps", type=int, default=25, required=True,
        help="FPS of the input and output videos.",
    )

class RemoveAnythingVideo(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target="ostrack",
            segmentor_target="sam",
            inpainter_target="sttn",
    ):
        super().__init__()
        tracker_build_args = {
            "tracker_param": args.tracker_ckpt
        }
        inpainter_build_args = {
            "lama": {
                "lama_config": args.lama_config,
                "lama_ckpt": args.lama_ckpt
            },
            "sttn": {
                "model_type": "sttn",
                "ckpt_p": args.vi_ckpt
            }
        }

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support sam now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="sttn", **kwargs):
        if target == "lama":
            return build_lama_model(**kwargs)
        elif target == "sttn":
            return build_sttn_model(**kwargs)
        else:
            raise NotImplementedError("Only support lama and sttn")

    def forward_tracker(self, frames_ps, init_box):
        init_box = np.array(init_box).astype(np.float32).reshape(-1, 4)
        seq = Sequence("tmp", frames_ps, 'inpaint-anything', init_box)
        all_box_xywh = get_box_using_ostrack(self.tracker, seq)
        return all_box_xywh

    def forward_segmentor(self, img, point_coords=None, point_labels=None,
                          box=None, mask_input=None, multimask_output=True,
                          return_logits=False):
        self.segmentor.set_image(img)

        masks, scores, logits = self.segmentor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, frames, masks):
        print(self.inpainter_target)
        if self.inpainter_target == "lama":
            for idx in range(len(frames)):
                frames[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target == "sttn":
            frames = [Image.fromarray(frame) for frame in frames]
            masks = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            frames = inpaint_video_with_builded_sttn(
                self.inpainter, frames, masks, device=self.device)
        else:
            raise NotImplementedError
        return frames

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def mask_selection(self, masks, scores, ref_mask=None, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            if ref_mask is not None:
                mse = np.mean(
                    (masks.astype(np.int32) - ref_mask.astype(np.int32))**2,
                    axis=(-2, -1)
                )
                idx = mse.argmin()
            else:
                idx = scores.argmax()
            return masks[idx]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return np.array([x, y, w, h])

    def forward(
            self,
            frame_ps: List[str],
            key_frame_idx: int,
            key_frame_point_coords: np.ndarray,
            key_frame_point_labels: np.ndarray,
            key_frame_mask_idx: int = None,
            dilate_kernel_size: int = 15,
    ):
        """
        Mask is 0-1 ndarray in default
        Frame is 0-255 ndarray in default
        """
        assert key_frame_idx == 0, "Only support key frame at the beginning."

        # get key-frame mask
        key_frame_p = frame_ps[key_frame_idx]
        key_frame = iio.imread(key_frame_p)
        key_masks, key_scores = self.forward_segmentor(
            key_frame, key_frame_point_coords, key_frame_point_labels)

        # key-frame mask selection
        if key_frame_mask_idx is not None:
            key_mask = key_masks[key_frame_mask_idx]
        else:
            key_mask = self.mask_selection(key_masks, key_scores)
        
        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)

        # get key-frame box
        key_box = self.get_box_from_mask(key_mask)

        # get all-frame boxes using video tracker
        print("Tracking ...")
        all_box = self.forward_tracker(frame_ps, key_box)

        # get all-frame masks using sam
        print("Segmenting ...")
        all_mask = [key_mask]
        all_frame = [key_frame]
        ref_mask = key_mask
        for frame_p, box in zip(frame_ps[1:], all_box[1:]):
            frame = iio.imread(frame_p)

            # XYWH -> XYXY
            x, y, w, h = box
            sam_box = np.array([x, y, x + w, y + h])
            masks, scores = self.forward_segmentor(frame, box=sam_box)
            mask = self.mask_selection(masks, scores, ref_mask)
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)
            # print(mask.shape)
            ref_mask = mask
            all_mask.append(mask)
            all_frame.append(frame)

        # get all-frame inpainted results
        print("Inpainting ...")
        all_frame = self.forward_inpainter(all_frame, all_mask)
        return all_frame, all_mask, all_box


def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels,
                size=(width * 0.04) ** 2)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_box(img, box):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(width / dpi / 0.77, height / dpi / 0.77))
    ax.imshow(img)
    ax.axis('off')

    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp(".png")
    fig.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    # inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RemoveAnythingVideo(args)
    model.to(device)
    
    masks = np.load('./chatsim/masks.npy', allow_pickle = True)
    frames = np.load('./chatsim/current_images.npy', allow_pickle = True)
    
    with torch.no_grad():
        all_frame_rm_w_mask = model.forward_inpainter(frames, masks)

    np.save('./chatsim/inpainted_imgs.npy', all_frame_rm_w_mask)
