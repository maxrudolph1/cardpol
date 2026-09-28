import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('relu'))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    """DrQ-v2 conv trunk + linear projection to `feature_dim`.

    Input: a frame stack (B, 3 * frame_stack, 84, 84) scaled to [-1, 1].
    output_norm='layernorm' (LayerNorm + tanh) is used by CARDPol; 'none' by polcon.
    """

    def __init__(self, in_channels=9, feature_dim=256, output_norm='layernorm', img_size=84):
        super().__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        self.convnet.apply(weight_init)  # the projection keeps PyTorch's default init
        conv_dim = self.convnet(torch.zeros(1, in_channels, img_size, img_size)).numel()
        projection = [nn.Linear(conv_dim, feature_dim)]
        if output_norm == 'layernorm':
            projection += [nn.LayerNorm(feature_dim), nn.Tanh()]
        elif output_norm != 'none':
            raise ValueError(f'unknown output_norm {output_norm}')
        self.projection = nn.Sequential(*projection)
        self.repr_dim = feature_dim

    def forward(self, obs):
        return self.projection(self.convnet(obs.float()).flatten(1))


class MLP(nn.Module):
    """`num_layers` hidden ReLU layers of width `hidden_dim`."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
                                    + [nn.Linear(dims[-1], out_dim)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
