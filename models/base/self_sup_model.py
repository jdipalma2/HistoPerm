import pytorch_lightning as pl
from torchvision import datasets, transforms
from utilities import utils
import torch
from typing import Any
import math
from torch.utils.data import DataLoader
from utilities.lars import LARS


class SelfSupModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters(params)

        # Create dataset
        self.view_1_data_transform = utils.DataAugmentation(cj_brightness=self.hparams.view_1_cj_brightness,
                                                            cj_contrast=self.hparams.view_1_cj_contrast,
                                                            cj_hue=self.hparams.view_1_cj_hue,
                                                            cj_prob=self.hparams.view_1_cj_prob,
                                                            cj_saturation=self.hparams.view_1_cj_saturation,
                                                            gauss_blur_divider=self.hparams.view_1_gauss_blur_divider,
                                                            gauss_prob=self.hparams.view_1_gauss_prob,
                                                            gauss_sigma=self.hparams.view_1_gauss_sigma,
                                                            gs_prob=self.hparams.view_1_gs_prob,
                                                            mean=self.hparams.mean,
                                                            patch_size=self.hparams.crop_size,
                                                            solarize_prob=self.hparams.view_1_solarize_prob,
                                                            solarize_threshold=self.hparams.view_1_solarize_threshold,
                                                            std=self.hparams.std,
                                                            crop_prob=self.hparams.view_1_crop_prob,
                                                            hor_flip_prob=self.hparams.view_1_hor_flip_prob,
                                                            vert_flip_prob=self.hparams.view_1_vert_flip_prob)
        self.view_2_data_transform = utils.DataAugmentation(cj_brightness=self.hparams.view_2_cj_brightness,
                                                            cj_contrast=self.hparams.view_2_cj_contrast,
                                                            cj_hue=self.hparams.view_2_cj_hue,
                                                            cj_prob=self.hparams.view_2_cj_prob,
                                                            cj_saturation=self.hparams.view_2_cj_saturation,
                                                            gauss_blur_divider=self.hparams.view_2_gauss_blur_divider,
                                                            gauss_prob=self.hparams.view_2_gauss_prob,
                                                            gauss_sigma=self.hparams.view_2_gauss_sigma,
                                                            gs_prob=self.hparams.view_2_gs_prob,
                                                            mean=self.hparams.mean,
                                                            patch_size=self.hparams.crop_size,
                                                            solarize_prob=self.hparams.view_2_solarize_prob,
                                                            solarize_threshold=self.hparams.view_2_solarize_threshold,
                                                            std=self.hparams.std,
                                                            crop_prob=self.hparams.view_2_crop_prob,
                                                            hor_flip_prob=self.hparams.view_2_hor_flip_prob,
                                                            vert_flip_prob=self.hparams.view_2_vert_flip_prob)

        # Data should have "train" subfolder
        self.train_dataset = datasets.ImageFolder(root=str(self.hparams.data_path.joinpath("train")),
                                                  transform=transforms.ToTensor())

        # Parameters related to learning rate scaling.
        self.warmup_steps = int(
            round((self.hparams.warmup_epochs * len(self.train_dataset)) // self.hparams.batch_size))
        self.max_iteration = (len(self.train_dataset) * self.hparams.max_epochs) // (self.hparams.batch_size + 1)

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        optimizer = self.optimizers()
        if self.global_step < self.warmup_steps:
            learning_rate = (self.global_step / self.warmup_steps) * self.hparams.lr if self.warmup_steps else self.hparams.lr
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate
        else:
            global_step = min(self.global_step - self.warmup_steps, self.max_iteration - self.warmup_steps)
            cosine_decay = self.hparams.lr * 0.5 * (1.0 + math.cos(math.pi * global_step / (self.max_iteration - self.warmup_steps)))
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_decay

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []

        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue
            if "classifier" in name:
                continue
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {"params": excluded_parameters, "names": excluded_parameter_names, "use_lars": False, "weight_decay": 0}
        ]

        encoding_optimizer = LARS(param_groups, lr=self.hparams.lr, momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay, eta=self.hparams.eta)
        encoding_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoding_optimizer, gamma=1.0)

        return [encoding_optimizer], [encoding_scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_data_workers, pin_memory=self.hparams.pin_data_memory,
                          drop_last=self.hparams.drop_last_batch, shuffle=True)
