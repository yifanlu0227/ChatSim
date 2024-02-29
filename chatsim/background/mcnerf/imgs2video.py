import cv2
import os
import numpy as np
from PIL import Image
import imageio

def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    # im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))
    im_list = sorted([os.path.join(im_dir, img) for img in os.listdir(im_dir) if img.endswith((".png", ".jpg", ".jpeg"))])
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # videoWriter = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, img_size)
    


    for i in im_list:
        im_name = os.path.join(im_dir, i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)



    videoWriter.release()
    print('Done')


if __name__ == '__main__':
    img_folder = '/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/exp/segment-17761959194352517553_5448_420_5468_420_with_camera_labels/exp_1029/novel_images'  # 替换为你的图片文件夹路径
    output_video = '/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/exp/segment-17761959194352517553_5448_420_5468_420_with_camera_labels/exp_1029/novel_images/1776_output_exp_1029_rotate.mp4'  # 你希望保存的视频文件路径
    fps = 2

    frame2video(img_folder, output_video, fps)
