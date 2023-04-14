import os
import numpy as np

from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image


def scale(val):
    return (val * 2) - 1


def to_tensor(data):
    return torch.tensor(data, device="cuda" if torch.cuda.is_available() else "cpu")


# Randomized Fourier Features improve rendering quality
def random_fourier_features(coord, dim):
    coord = np.expand_dims(coord, axis=1)
    weights = np.expand_dims(np.random.rand(dim), axis=0)
    freqs = coord * weights * 2 * np.pi

    sin = np.sin(freqs.T)
    cos = np.cos(freqs.T)
    return sin, cos


class NeRFDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()

        # Load and scale image
        img = np.array(Image.open(config.inp_path))
        img = img / 255.0

        H, W, C = img.shape

        x_sin, x_cos = [], []
        y_sin, y_cos = [], []

        # Get relative coordinates in range (-1, 1)
        x = scale(np.linspace(0, W - 1, num=W) / W)
        y = scale(np.linspace(0, H - 1, num=H) / H)

        # Pre-compute pixel values -- reduces overall complexity of init
        if config.use_positional_encoding:
            # Using fourier features
            for exp in range(config.max_freq_exp):
                freq = (2**exp) * np.pi

                x_sin.append(np.sin(freq * x))
                x_cos.append(np.cos(freq * x))

                y_sin.append(np.sin(freq * y))
                y_cos.append(np.cos(freq * y))
        elif config.use_random_fourier_features:
            x_sin, x_cos = random_fourier_features(x, config.max_freq_exp)
            y_sin, y_cos = random_fourier_features(y, config.max_freq_exp)

        self.pixel_encoding = []
        self.color_space = []

        for i in range(H):
            for j in range(W):
                # Scaled color space
                r, g, b = img[i, j]
                r = scale(r)
                g = scale(g)
                b = scale(b)

                positions = []
                if (
                    not config.use_positional_encoding
                    and not config.use_random_fourier_features
                ):
                    y_pos = scale(i / H)
                    x_pos = scale(j / W)

                    positions.append(x_pos)
                    positions.append(y_pos)
                else:
                    for exp in range(config.max_freq_exp):
                        positions.append(x_sin[exp][j])
                        positions.append(x_cos[exp][j])

                        positions.append(y_sin[exp][i])
                        positions.append(y_cos[exp][i])

                self.pixel_encoding.append(np.array(positions))
                self.color_space.append(np.array([r, g, b]))

        assert len(self.pixel_encoding) == len(
            self.color_space
        ), f"Input ({len(self.pixel_encoding)}) and output ({len(self.color_space)}) size mismatch"

    def __len__(self):
        return len(self.pixel_encoding)

    def __getitem__(self, idx):
        pixel_sample = to_tensor(self.pixel_encoding[idx])
        color_sample = to_tensor(self.color_space[idx])

        batch = {
            "input_coord": pixel_sample,
            "output_color": color_sample,
        }

        return batch


def get_dataloader(
    config,
    num_workers,
    training=True,
    batch_size=None,
):
    num_gpus = config.devices if training else 1
    data = NeRFDataset(config)

    data[len(data) - 1]  # Check to prevent index errors

    return DataLoader(
        dataset=data,
        batch_size=batch_size or config.batch_size,
        shuffle=training,
        num_workers=num_workers
        if num_workers is not None
        else int(cpu_count() / num_gpus),
        worker_init_fn=None,
        drop_last=training,
        pin_memory=not training,
    )
