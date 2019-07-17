
import functools

import imlib as im
import numpy as np
import pylib as py
import tensorboardX
import torch
import torchlib
import torchprob as gan
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
py.arg('--dataset', default='fashion_mnist', choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=25)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--z_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'dragan-lp', 'wgan-gp', 'wgan-lp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='none')
py.arg('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])  # !!!
args = py.args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s' % args.gradient_penalty_mode
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    data_loader, shape = data.make_32x32_dataset(args.dataset, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset == 'celeba':  # 64x64
    img_paths = py.glob('data/img_align_celeba', '*.jpg')
    data_loader, shape = data.make_celeba_dataset(img_paths, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'anime':  # 64x64
    img_paths = py.glob('data/faces', '*.jpg')
    data_loader, shape = data.make_anime_dataset(img_paths, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'custom':
    # ======================================
    # =               custom               =
    # ======================================
    img_paths = ...  # image paths of custom dataset
    data_loader = data.make_custom_dataset(img_paths, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = ...  # 3 for 32x32 and 4 for 64x64
    # ======================================
    # =               custom               =
    # ======================================


# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = module.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = module.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
print(D)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def train_G():
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return {'g_loss': G_loss}


def train_D(x_real):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp = gan.gradient_penalty(functools.partial(D), x_real, x_fake, mode=args.gradient_penalty_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# load checkpoint if exists
ckpt_dir = py.join(output_dir, 'checkpoints')
py.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    D_optimizer.load_state_dict(ckpt['D_optimizer'])
    G_optimizer.load_state_dict(ckpt['G_optimizer'])
except:
    ep, it_d, it_g = 0, 0, 0

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(py.join(output_dir, 'summaries'))
z = torch.randn(100, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

for ep_ in tqdm.trange(args.epochs, desc='Epoch Loop'):
    if ep_ < ep:
        continue
    ep += 1

    # train for an epoch
    for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
        x_real = x_real.to(device)

        D_loss_dict = train_D(x_real)
        it_d += 1
        for k, v in D_loss_dict.items():
            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

        if it_d % args.n_d == 0:
            G_loss_dict = train_G()
            it_g += 1
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

        # sample
        if it_g % 100 == 0:
            x_fake = sample(z)
            x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))
            img = im.immerge(x_fake, n_rows=10).squeeze()
            im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % it_g))

    # save checkpoint
    torchlib.save_checkpoint({'ep': ep, 'it_d': it_d, 'it_g': it_g,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'D_optimizer': D_optimizer.state_dict(),
                              'G_optimizer': G_optimizer.state_dict()},
                             py.join(ckpt_dir, 'Epoch_(%d).ckpt' % ep),
                             max_keep=1)
