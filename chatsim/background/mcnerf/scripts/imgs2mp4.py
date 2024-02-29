import click
import os
import numpy as np
import cv2 as cv
from glob import glob
from os.path import join as pjoin
import ipdb
import imageio 

@click.command()
@click.option('--data_dir', type=str)
@click.option('--suffix', type=str, default='*.png')
@click.option('--fps', type=int, default=20)

def hello(data_dir, suffix, fps):
    # image_list = sorted(glob(pjoin(data_dir, suffix)))
    input_dir = '/dssg/home/acct-umjpyb/umjpyb/ziwang/edition_system/outputs/1137_demo2_120frames_2023_11_22_13_52_57/__init__/'
    image_list = []
    for i in range(100):
        image_list.append(input_dir+str(i)+'.png')

    imgs = []
    for img_path in image_list:
        imgs.append(cv.imread(img_path)[:,:])

    last = imgs[-1]
    for i in range(30):
        imgs.append(last)

    height, width, layers = imgs[-1].shape
    size = (width, height)
    # imgs = imgs[::-1]
    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # out = cv.VideoWriter(pjoin(data_dir, 'output.mp4'), fourcc, fps, size, True)
    # for img in imgs:
    #     out.write(img)



    # out.release()
    writer = imageio.get_writer(pjoin(data_dir, 'output.mp4'), 
                                    fps=fps)
    for i, frame in enumerate(imgs):

        writer.append_data(np.array(frame[:,:,::-1]))
    writer.close()


if __name__ == '__main__':
    hello()

