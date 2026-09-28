"""Multi-policy trajectory windows for representation learning."""
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MultiPolicyDataset(Dataset):
    """Windows of `horizon` consecutive steps from one episode, labelled by the policy that produced them.

    paths[i] holds the rollouts of policy i, so the label is the index into `paths`.
    Items: frames uint8 (horizon, 3, 84, 84), state float32 (horizon, state_dim), label int64.
    """

    def __init__(self, paths, horizon):
        self.horizon = horizon
        self.frames, self.states, self.index = [], [], []
        for label, path in enumerate(paths):
            print(f'loading {path}', flush=True)
            with h5py.File(path, 'r') as f:
                self.frames.append(f['frames'][:])
                self.states.append(f['state'][:].astype(np.float32))
                episode = f['episode_index'][:]
            # a window is valid if its first and last steps are in the same episode
            starts = np.nonzero(episode[:len(episode) - horizon + 1] == episode[horizon - 1:])[0]
            self.index.append(np.stack([np.full_like(starts, label), starts], axis=1))
        self.index = np.concatenate(self.index)
        states = np.concatenate(self.states)
        self.state_mean, self.state_std = states.mean(0), np.sqrt(states.var(0) + 1e-8)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        label, start = self.index[i]
        window = slice(start, start + self.horizon)
        return {'frames': torch.from_numpy(self.frames[label][window]).permute(0, 3, 1, 2),
                'state': torch.from_numpy(self.states[label][window]),
                'label': torch.tensor(label)}


def make_loader(dataset, batch_size, num_workers, shuffle=True):
    kwargs = dict(persistent_workers=True, prefetch_factor=2) if num_workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, drop_last=True, **kwargs)


def to_unit_range(frames):
    """uint8 [0, 255] -> float [-1, 1]."""
    return frames.float() / 255.0 * 2.0 - 1.0


def augment(frames, crop_size=80):
    """Random crop (resized back to 84) + brightness/contrast/saturation jitter, drawn per frame.

    frames: (B, T, C, 84, 84) in [-1, 1].
    """
    shape = frames.shape
    x = (frames.reshape(-1, shape[2], 84, 84) + 1.0) / 2.0
    n = x.shape[0]
    max_offset = 84 - crop_size
    top = torch.randint(0, max_offset + 1, (n,), device=x.device)
    left = torch.randint(0, max_offset + 1, (n,), device=x.device)
    # batched crop + resize with a single affine grid_sample
    theta = torch.zeros((n, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = crop_size / 84.0
    theta[:, 0, 2] = 2.0 * (left.float() + crop_size / 2.0) / 84.0 - 1.0
    theta[:, 1, 2] = 2.0 * (top.float() + crop_size / 2.0) / 84.0 - 1.0
    grid = F.affine_grid(theta, size=(n, x.shape[1], 84, 84), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    brightness, contrast, saturation = (torch.empty((n, 1, 1, 1), device=x.device).uniform_(0.9, 1.1)
                                        for _ in range(3))
    x = x * brightness
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    x = ((x - gray) * saturation + gray).clamp(0.0, 1.0)
    return (x * 2.0 - 1.0).reshape(shape)
