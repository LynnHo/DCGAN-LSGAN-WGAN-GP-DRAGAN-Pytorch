from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils


def gradient_penalty(x, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = utils.cuda(torch.rand(shape))
    beta = utils.cuda(torch.rand(x.size()))

    y = x + 0.5 * x.std() * beta
    z = x + alpha * (y - x)

    # gradient penalty
    z = utils.cuda(Variable(z, requires_grad=True))
    o = f(z)
    g = grad(o, z, grad_outputs=utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp

""" gpu """
gpu_id = [2]
utils.cuda_devices(gpu_id)


""" param """
epochs = 50
batch_size = 64
lr = 0.0002
z_dim = 100


""" data """
crop_size = 108
re_size = 64
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(crop),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

imagenet_data = dsets.ImageFolder('./data/img_align_celeba', transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)


""" model """
D = models_64x64.DiscriminatorWGANGP(3)
G = models_64x64.Generator(z_dim)
bce = nn.BCEWithLogitsLoss()
utils.cuda([D, G, bce])

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


""" load checkpoint """
ckpt_dir = './checkpoints/celeba_dragan'
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


""" run """
writer = tensorboardX.SummaryWriter('./summaries/celeba_dragan')

z_sample = Variable(torch.randn(100, z_dim))
z_sample = utils.cuda(z_sample)
for epoch in range(start_epoch, epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable(imgs)
        bs = imgs.size(0)
        z = Variable(torch.randn(bs, z_dim))
        r_lbl = Variable(torch.ones(bs))
        f_lbl = Variable(torch.zeros(bs))
        imgs, z, r_lbl, f_lbl = utils.cuda([imgs, z, r_lbl, f_lbl])

        f_imgs = G(z)

        # train D
        r_logit = D(imgs)
        f_logit = D(f_imgs.detach())
        d_r_loss = bce(r_logit, r_lbl)
        d_f_loss = bce(f_logit, f_lbl)
        gp = gradient_penalty(imgs.data, D)
        d_loss = d_r_loss + d_f_loss + gp * 10.0

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalars('D/loss',
                           {"d_loss": d_loss.data.cpu().numpy(),
                            "d_r_loss": d_r_loss.data.cpu().numpy(),
                            "d_f_loss": d_f_loss.data.cpu().numpy()},
                           global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        # train G
        f_logit = D(f_imgs)
        g_loss = bce(f_logit, r_lbl)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        writer.add_scalars('G',
                           {"g_loss": g_loss.data.cpu().numpy()},
                           global_step=step)

        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(data_loader)))

        if (i + 1) % 100 == 0:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2.0

            save_dir = './sample_images_while_training/celeba_dragan'
            utils.mkdir(save_dir)
            torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)

    utils.save_checkpoint({'epoch': epoch + 1,
                           'D': D.state_dict(),
                           'G': G.state_dict(),
                           'd_optimizer': d_optimizer.state_dict(),
                           'g_optimizer': g_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)
