# WGAN-GP and DRAGAN

Pytorch implementation of WGAN-GP and DRAGAN, both of which use gradient penalty to enhance the training quality. We use DCGAN as the network architecture in all experiments.

WGAN-GP: [Improved Training of Wasserstein GANs](http://arxiv.org/abs/1704.00028)

DRAGAN: [How to Train Your DRAGAN](https://arxiv.org/abs/1705.07215)

## Exemplar results

### Celeba
left: WGAN-GP 50 epoch, right: DRAGAN 50 epoch

<img src="./pics/celeba_wgan_50.jpg" width="48%" height="48%"> <img src="./pics/celeba_wgan_gp_50.jpg" width="48%" height="48%">


# Prerequisites
- pytorch 0.2
- tensorboardX https://github.com/lanpa/tensorboard-pytorch
- python 2.7

# Usage

## Train
```
train_celeba_wgan_gp.py
train_celeba_dragan.py
...
```
## Tensorboard
If you have installed tensorboard, you can use it to have a look at the loss curves.
```
tensorboard --logdir=./summaries/celeba_wgan_gp --port=6006
...
```

## Datasets
1. Celeba should be prepared by yourself in ***./data/img_align_celeba/img_align_celeba/***
    - Download the dataset: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0
