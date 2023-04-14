import os
import logging

import numpy as np
import lightning.pytorch as pl
import torch
from torch import optim
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger

from PIL import Image

from .dataloader.definition import get_dataloader
from .model.definition import NeRF

from .utils.parser import ArgumentParser
from .utils.opts import *

logger = logging.getLogger("nerf2Dtoy")
logger.addHandler(logging.StreamHandler())


class NeRFModule(pl.LightningModule):
    """NeRF Lightning Module
    Args:
        config (ArgumentParser): Config object
        model (NeRF): NeRF model
    """

    def __init__(self, config, model) -> None:
        super(NeRFModule, self).__init__()

        self.config = config
        self.model = model
        self.validation_step_outputs = []

        # cache shape for validation image
        img = np.array(Image.open(self.config.inp_path))
        self.H, self.W, self.C = img.shape

    def training_step(self, batch, batch_idx):
        input_coord = batch["input_coord"]
        orig_color = batch["output_color"]

        # Convert doubles to float32
        input_coord = input_coord.to(torch.float32)
        orig_color = orig_color.to(torch.float32)

        # Forward pass
        gen_color = self.model(input_coord)
        loss = F.mse_loss(gen_color, orig_color)

        self.log_dict({f"MSE": loss}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def train_dataloader(self):
        return get_dataloader(self.config, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return get_dataloader(
            self.config,
            num_workers=self.config.num_workers,
            training=False,
            batch_size=1,
        )

    def validation_step(self, batch, batch_idx):
        input_coord = batch["input_coord"]
        input_coord = input_coord.to(torch.float32)
        gen_color = self.model(input_coord)
        self.validation_step_outputs.append(gen_color)
        return gen_color

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)  # (num_pixels, 3)

        # Switch to tackle pl sanity check
        if all_preds.shape[0] < (self.H * self.W):
            self.validation_step_outputs.clear()  # free memory
            pass
        else:
            img = torch.zeros(self.C, self.H, self.W)
            for i in range(self.H):
                img[:, i, :] = all_preds[
                    self.W * i : self.W * (i + 1), :
                ].T  # Shape expected by logger (C, H, W)
            self.validation_step_outputs.clear()  # free memory

            # Convert output to image and log to tensorboard
            img = img.detach().cpu().numpy()
            img[0, :, :] = np.clip((img[0, :, :] + 1) / 2.0, 0, 1)
            img[1, :, :] = np.clip((img[1, :, :] + 1) / 2.0, 0, 1)
            img[2, :, :] = np.clip((img[2, :, :] + 1) / 2.0, 0, 1)
            img = (img * 255.0).astype(np.uint8)

            self.logger.experiment.add_image("Neural Rendering", img, self.global_step)


def _get_parser():
    parser = ArgumentParser(description="train.py")

    model_opts(parser)
    data_opts(parser)
    trainer_opts(parser)

    return parser


def _trainer_init(config):
    """
    Initialize trainer callbacks and logger
    """
    expt_name = os.path.basename(config.save_dir)
    log_dir = config.log_dir

    logger = TensorBoardLogger(log_dir, expt_name)

    ckpt = ModelCheckpoint(
        config.save_dir,
        every_n_epochs=config.save_every,
        every_n_train_steps=None,
        save_last=config.save_last,
        save_top_k=1,
    )

    lrm = LearningRateMonitor("step", log_momentum=True)
    summary = ModelSummary(-1)
    callbacks = [ckpt, summary, lrm]

    return callbacks, logger


def main():
    parser = _get_parser()
    config = parser.parse_args()

    # Define training callbacks
    callbacks, logger = _trainer_init(config)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
    )
    inp_size = (
        2
        if not config.use_positional_encoding
        and not config.use_random_fourier_features
        and not config.learnable_positional_encoding
        else (4 * config.max_freq_exp)
    )
    model = NeRFModule(config, NeRF(config, inp_size))
    trainer.fit(model=model)


if __name__ == "__main__":
    main()
