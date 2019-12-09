import torch


# ======================================
# =           sample method            =
# ======================================

def _sample_line(real, fake):
    shape = [real.size(0)] + [1] * (real.dim() - 1)
    alpha = torch.rand(shape, device=real.device)
    sample = real + alpha * (fake - real)
    return sample


def _sample_DRAGAN(real, fake):  # fake is useless
    beta = torch.rand_like(real)
    fake = real + 0.5 * real.std() * beta
    sample = _sample_line(real, fake)
    return sample


# ======================================
# =      gradient penalty method       =
# ======================================

def _norm(x):
    norm = x.view(x.size(0), -1).norm(p=2, dim=1)
    return norm


def _one_mean_gp(grad):
    norm = _norm(grad)
    gp = ((norm - 1)**2).mean()
    return gp


def _zero_mean_gp(grad):
    norm = _norm(grad)
    gp = (norm**2).mean()
    return gp


def _lipschitz_penalty(grad):
    norm = _norm(grad)
    gp = (torch.max(torch.zeros_like(norm), norm - 1)**2).mean()
    return gp


def gradient_penalty(f, real, fake, gp_mode, sample_mode):
    sample_fns = {
        'line': _sample_line,
        'real': lambda real, fake: real,
        'fake': lambda real, fake: fake,
        'dragan': _sample_DRAGAN,
    }

    gp_fns = {
        '1-gp': _one_mean_gp,
        '0-gp': _zero_mean_gp,
        'lp': _lipschitz_penalty,
    }

    if gp_mode == 'none':
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    else:
        x = sample_fns[sample_mode](real, fake).detach()
        x.requires_grad = True
        pred = f(x)
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        gp = gp_fns[gp_mode](grad)

    return gp
