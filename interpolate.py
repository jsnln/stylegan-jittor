import os
import tqdm
import argparse
import math

import jittor as jt
jt.flags.use_cuda = 1
# from torchvision import utils

from model import StyledGenerator


@jt.no_grad()
def get_mean_style(generator):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(jt.randn(1024, 512))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@jt.no_grad()
def sample(generator, step, mean_style, n_sample):
    image = generator(
        jt.randn(n_sample, 512),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

@jt.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target):
    source_code = jt.randn(n_source, 512)
    target_code = jt.randn(n_target, 512)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [jt.ones([1, 3, shape, shape]) * -1]

    source_image = generator(source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)
    target_image = generator(target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = jt.concat(images, 0)
    
    return images

@jt.no_grad()
def gen_code_list(n_codes_interp, n_codes_perimg):
    code_list = []
    for i in range(n_codes_interp):
        code_list.append(jt.randn(n_codes_perimg, 512))
    return code_list

@jt.no_grad()
def style_interpolate(generator, step, mean_style, source_code, target_code, interp_steps=20, return_last=False):
    # shape = 4 * 2 ** step
    alpha = 1

    images = []
    for T in range(interp_steps+1 if return_last else interp_steps):
        t = T / interp_steps
        code = source_code * (1 - t) + target_code * t
        image = generator(code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)
        images.append(image)

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--expname', type=str, required=True, help='experiment folder')
    parser.add_argument('--size', type=int, default=128, help='size of the image')
    parser.add_argument('--n_row', type=int, default=5, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=10, help='number of columns of sample matrix')
    parser.add_argument('--code_size', type=int, default=512, help='latent code size')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()
    
    generator = StyledGenerator(args.code_size)
    generator.load_state_dict(jt.load(args.path))
    generator.eval()

    mean_style = get_mean_style(generator)
    step = int(math.log(args.size, 2)) - 2

    n_codes = 101
    interp_steps = 20
    code_list = gen_code_list(n_codes, args.n_row * args.n_col)
    imgs_list = []
    print(f"[LOG] Generating {(n_codes-1) * interp_steps} images...")
    for i in tqdm.tqdm(range(len(code_list)-1)):
        source_code = code_list[i]
        target_code = code_list[i+1]
        return_last = False if i < len(code_list)-2 else True
        imgs = style_interpolate(generator, step, mean_style, source_code, target_code, interp_steps=interp_steps, return_last=return_last)
        imgs_list += imgs
    
    if not os.path.exists(f'{args.expname}/interpolation/'):
        os.makedirs(f'{args.expname}/interpolation/')
    print(f"[LOG] Saving interpolated images...")
    for i, img in enumerate(tqdm.tqdm(imgs_list)):
        jt.misc.save_image(
            img, f'{args.expname}/interpolation/interpolation_{i:05d}.png', nrow=args.n_col, normalize=True, range=(-1, 1)
        )
