import os
import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import numpy as np
from einops import rearrange
import cv2

from PIL import Image
import lmdb
from tqdm import tqdm
import jittor


def resize_and_convert(img, size, quality=100):
    img = jittor.transform.resize(img, size, Image.LANCZOS)
    img = jittor.transform.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val

def resize_multiple(img, sizes, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs

def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(transaction, dataset, n_worker, sizes):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    
    args = parser.parse_args()

    if args.path.upper() == 'MNIST':
        print(f"Using built-in MNIST dataset. Max size = 32.")
        def create_mnist_datafolder():
            mnist_dataset = jittor.dataset.MNIST()
            for i in range(10):
                os.makedirs(f'MNIST/{i:02d}', exist_ok=True)
            for i in range(len(mnist_dataset)):
                img = rearrange(mnist_dataset[i][0], 'c h w -> h w c')
                img = (img * 255).astype(np.uint8)
                num = mnist_dataset[i][1]
                cv2.imwrite(f'MNIST/{num:02d}/{i:04d}.png', img)
        create_mnist_datafolder()
        imgset = jittor.dataset.ImageFolder(args.path)
        sizes = (8, 16, 32)
    elif os.path.exists(args.path):
        imgset = jittor.dataset.ImageFolder(args.path)
        sizes = (8, 16, 32, 64, 128, 256, 512, 1024)
    else:
        print(f"[ERROR] cannot find dataset {args.path}")
        exit()
        
    # print(imgset)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker, sizes=sizes)
