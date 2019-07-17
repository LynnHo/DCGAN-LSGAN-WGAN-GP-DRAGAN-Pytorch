import torch


def get_gan_losses_fn():
    bce = torch.nn.BCEWithLogitsLoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(r_logit, torch.ones_like(r_logit))
        f_loss = bce(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = torch.nn.MSELoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(r_logit, torch.ones_like(r_logit))
        f_loss = mse(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = -r_logit.mean()
        f_loss = f_logit.mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


def gradient_penalty(f, real, fake, mode, p_norm=2):
    def _gradient_penalty(f, real, fake=None, penalty_type='gp', p_norm=2):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = torch.rand_like(a)
                b = a + 0.5 * a.std() * beta
            shape = [a.size(0)] + [1] * (a.dim() - 1)
            alpha = torch.rand(shape, device=a.device)
            inter = a + alpha * (b - a)
            return inter

        x = _interpolate(real, fake).detach()
        x.requires_grad = True
        pred = f(x)
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        norm = grad.view(grad.size(0), -1).norm(p=p_norm, dim=1)

        if penalty_type == 'gp':
            gp = ((norm - 1)**2).mean()
        elif penalty_type == 'lp':
            gp = (torch.max(torch.zeros_like(norm), norm - 1)**2).mean()

        return gp

    if mode == 'none':
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    elif mode in ['dragan', 'dragan-gp', 'dragan-lp']:
        penalty_type = 'gp' if mode == 'dragan' else mode[-2:]
        gp = _gradient_penalty(f, real, penalty_type=penalty_type, p_norm=p_norm)
    elif mode in ['wgan-gp', 'wgan-lp']:
        gp = _gradient_penalty(f, real, fake, penalty_type=mode[-2:], p_norm=p_norm)

    return gp
