***Recommendation***

- Our GAN based work for facial attribute editing - [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow).

***News***

- 28 June 2019: We re-implement these GANs by **Pytorch 1.1**! The old version is here: [v0](https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch/tree/v0) or in the "v0" directory.
- [**Tensorflow 2** Version](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2)

<hr style="height:1px" />

<p align="center">
    <img src="./pics/celeba_dragan.gif" width="49.7%" />  <img src="./pics/anime_dragan.gif" width="49.7%" />
</p>

<hr style="height:1px" />

# <p align="center"> GANs - Pytorch </p>

Pytorch implementations of [DCGAN](https://arxiv.org/abs/1511.06434), [LSGAN](https://arxiv.org/abs/1611.04076), [WGAN-GP](http://arxiv.org/abs/1704.00028)([LP](https://arxiv.org/abs/1709.08894)) and [DRAGAN](https://arxiv.org/abs/1705.07215v5).

## Exemplar results

### Fashion-MNIST

DCGAN                                    | LSGAN                                      | WGAN-GP                                      | DRAGAN
:---:                                    | :---:                                      | :---:                                        | :---:
<img src="./pics/fashion-mnist_gan.jpg"> | <img src="./pics/fashion-mnist_lsgan.jpg"> | <img src="./pics/fashion-mnist_wgan-gp.jpg"> | <img src="./pics/fashion-mnist_dragan.jpg">

### CelebA

DCGAN                                 | LSGAN
:---:                                 | :---:
<img src="./pics/celeba_gan.jpg">     | <img src="./pics/celeba_lsgan.jpg">
**WGAN-GP**                           | **DRAGAN**
<img src="./pics/celeba_wgan-gp.jpg"> | <img src="./pics/celeba_dragan.jpg">
**WGAN-LP**                           | **DRAGAN-LP**
<img src="./pics/celeba_wgan-lp.jpg"> | <img src="./pics/celeba_dragan-lp.jpg">

### Anime

**WGAN-GP**                          | **DRAGAN**
:---:                                | :---:
<img src="./pics/anime_wgan-gp.jpg"> | <img src="./pics/anime_dragan.jpg">

# Usage

- Prerequisites

    - PyTorch 1.1
    - tensorboardX
    - scikit-image, oyaml, tqdm
    - Python 3.6

- Datasets

    - Fashion-MNIST will be automatically downloaded
    - CelebA should be prepared by yourself in **./data/img_align_celeba/\*.jpg**
        - dataset link: [Dropbox](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0)
        - the above link might be inaccessible, the alternatives are (find "img_align_celeba.zip")
            - [Baidu Netdisk](https://pan.baidu.com/s/1eSNpdRG#list/path=%2Fsharelink2785600790-938296576863897%2FCelebA%2FImg&parentPath=%2Fsharelink2785600790-938296576863897) or
            - [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)
    - the Anime dataset should be prepared by yourself in **./data/faces/\*.jpg**
        - dataset link: https://pan.baidu.com/s/1eSifHcA, password: g5qa
        - reference: https://zhuanlan.zhihu.com/p/24767059

- Examples of training

    - Fashion-MNIST DCGAN

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan
        ```

    - CelebA DRAGAN

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan
        ```

    - Anime WGAN-GP

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5
        ```

    - see more training exampls in [commands.sh](./commands.sh)

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir ./output/fashion_mnist_gan/summaries --port 6006
        ```

