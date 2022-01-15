import matplotlib.pyplot as plt
import argparse
import random
import math
import os
import time

from tqdm import tqdm
import jittor as jt
import jittor.transform as transform
from jittor import grad

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator

from icecream import ic

jt.flags.use_cuda = 1

def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    dataset.set_attrs(resolution=image_size, batch_size=batch_size)
    return dataset

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(args.max_iters))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:

        d_optimizer.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            jt.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'{args.expname}/checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        ### NOTE data loading
        try:
            real_image = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)
        
        used_sample += real_image.shape[0]
        b_size = real_image.size(0)

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            d_optimizer.backward(-real_predict)

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = jt.nn.softplus(-real_scores).mean()
            d_optimizer.backward(real_predict)

            grad_real = grad(real_scores.sum(), real_image)
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty
            d_optimizer.backward(grad_penalty)
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()

        ### NOTE generator forward
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, args.code_size).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, args.code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            d_optimizer.backward(fake_predict)

            eps = jt.rand(b_size, 1, 1, 1)
            
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(hat_predict.sum(), x_hat)

            grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            grad_penalty = 10 * grad_penalty
            d_optimizer.backward(grad_penalty)
            
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = jt.nn.softplus(fake_predict).mean()
            d_optimizer.backward(fake_predict)
            
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            g_optimizer.zero_grad()

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = jt.nn.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()

            g_optimizer.backward(loss)
            g_optimizer.step()

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        generator(
                            jt.randn(gen_j, args.code_size), step=step, alpha=alpha
                        ).data
                    )
            
            jt.misc.save_image(
                jt.concat(images, 0),
                f'{args.expname}/sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            jt.save(
                generator.state_dict(), f'{args.expname}/checkpoint/{str(i + 1).zfill(6)}.model'
            )
            jt.save(
                discriminator.state_dict(), f'{args.expname}/checkpoint/{str(i + 1).zfill(6)}.discrim-model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}; used: {used_sample}; bs: {dataset.batch_size}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    # code_size = 512
    # batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--expname', type=str, default='default_expname', help='name of the experiment, all data will be saved under this folder')
    parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
    parser.add_argument('--max_iters', default=3_000_000, type=int, help='max number of iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--no_from_rgb_activate', action='store_true', help='use activate in from_rgb (original implementation)')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'], help='class of gan loss')
    parser.add_argument('--batch_default', type=int, default=32, help='default batch (can be override by --sched')
    parser.add_argument('--code_size', type=int, default=512, help='latent code size')

    args = parser.parse_args()
    if not os.path.exists(args.expname):
        os.makedirs(os.path.join(args.expname, 'checkpoint'))
        os.makedirs(os.path.join(args.expname, 'sample'))


    generator = StyledGenerator(args.code_size)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)

    g_optimizer = jt.nn.Adam(generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        })
    d_optimizer = jt.nn.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))


    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)

        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transform.Compose(
        [
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
        # args.batch = {4: 512, 8: 256, 16: 128, 32: 32, 64: 16, 128: 16, 256: 16}  # works on single RTX3090

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    # args.batch_default = 32
    # args.batch_default = 16
    dataset = MultiResolutionDataset(args.path, transform)
    dataset.set_attrs(shuffle=True, drop_last=True, num_workers=0, endless=True)

    train(args, dataset, generator, discriminator)
