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


# ==============================================================================
# =                                layer wrapper                               =
# ==============================================================================

def identity(x, *args, **keyword_args):
    return x
