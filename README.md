# stylegan-jittor
A jittor version of StyleGAN. This code is modified from https://github.com/rosinality/style-based-gan-pytorch to support [jittor (a deep learning framework based on just-in-time compilation and meta-operators)](https://github.com/Jittor/Jittor).

### Preprocess Dataset

As in https://github.com/rosinality/style-based-gan-pytorch, the `lmdb` preprocessed dataset is used. First organize the picture as follows:

```bash
data-ffhq/
	ffhq/
		00000.png
		00001.png
		...
```

Then

```bash
python prepare_data.py data-ffhq --out data-ffhq-processed
```

### Train Command

#### FFHQ

```bash
python train.py data-ffhq-processed --loss r1 --sched --mixing --expname exp_ffhq_sched --max_size 128
```

### References

- StyleGAN: [official github repository](https://github.com/NVlabs/stylegan), [pytorch implementation](https://github.com/rosinality/style-based-gan-pytorch), [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)
- Jittor: [homepage](https://cg.cs.tsinghua.edu.cn/jittor/), [official github repository](https://github.com/Jittor/Jittor), [other jittor GANs](https://github.com/Jittor/gan-jittor)
