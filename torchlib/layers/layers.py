import torch


# ==============================================================================
# =                                    layer                                   =
# ==============================================================================

class Identity(torch.nn.Module):

    def __init__(self, *args, **keyword_args):
        super().__init__()

    def forward(self, x):
        return x


class Reshape(torch.nn.Module):

    def __init__(self, *new_shape):
        super().__init__()
        self._new_shape = new_shape

    def forward(self, x):
        new_shape = (x.size(i) if self._new_shape[i] == 0 else self._new_shape[i] for i in range(len(self._new_shape)))
        return x.view(*new_shape)


class DepthToSpace(torch.nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(torch.nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class ColorTransform(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, eps=1e-5):
        N, C, H, W = X.size()

        X = X.view(N, C, -1)  # (N, C, H*W)
        Y = Y.view(N, C, -1)  # (N, C, H*W)
        O = torch.ones(N, 1, H * W, dtype=X.dtype, device=X.device)  # (N, 1, H*W)
        X_ = torch.cat((X, O), dim=1)  # (N, C+1, H*W)
        X__T = X_.permute(0, 2, 1)  # (N, H*W, C+1)

        I = torch.eye(C + 1, dtype=X.dtype, device=X.device).view(-1, C + 1, C + 1).repeat([N, 1, 1])  # (N, C+1, C+1)
        A = Y.matmul(X__T).matmul((X_.matmul(X__T) + eps * I).inverse())  # (N, C, C+1)

        return A.matmul(X_).view(N, C, H, W)  # (N, C, H, W)


# ==============================================================================
# =                                layer wrapper                               =
# ==============================================================================

def identity(x, *args, **keyword_args):
    return x
