import os
import numpy as np
import cv2
import av
import glob
import tqdm
import matplotlib.pyplot as plt
import argparse
# import jittor as jt
# jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help="directory containing all images")
parser.add_argument('-o', '--output', type=str, required=True, help="output video file (no suffix, will add .mp4 automatically)")
parser.add_argument('-f', '--fps', type=int, default=40)
args = parser.parse_args()

FOLDER_PATH = args.directory
fps = args.fps
out_fn = './' + f'{args.output}.mp4'

png_list = sorted(list(glob.glob(FOLDER_PATH + '*.png')))

max_shape = cv2.imread(png_list[-1]).shape
h, w, _ = max_shape

# cv2 writer
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter(out_fn, fourcc, fps, (w, h))

for png_name in tqdm.tqdm(png_list):

    img = cv2.imread(png_name)
    if img.shape != max_shape:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    out.write(img)

out.release()
