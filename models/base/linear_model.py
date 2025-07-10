import math
from typing import Any

import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utilities import tn_resnet, utils


class LinearModel(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters(params)

        # actually do a load that is a little more flexible
        self.model = utils.get_encoder(self.hparams.encoder_arch)

        # Load the model from a checkpoint without certain keys.
        checkpoint = torch.load(self.hparams.ckpt_file)
        state_dict = checkpoint["state_dict"]
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_sd[k.replace("encoder.", "")] = v
            if k.startswith("online_encoder."):
                new_sd[k.replace("online_encoder.", "")] = v
        self.model.load_state_dict(new_sd, strict=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # Data
        self.train_transform = utils.LinearDataAugmentation(params=self.hparams)
        # Assume data directory has "train" and "val" subdirectories.
        self.train_dataset = datasets.ImageFolder(root=str(self.hparams.data_path.joinpath("train")),
                                                  transform=transforms.ToTensor())
        self.val_dataset = datasets.ImageFolder(root=str(self.hparams.data_path.joinpath("val")),
                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std)]))

        # Parameters related to learning rate scaling.
        self.warmup_steps = int(
            round((self.hparams.warmup_epochs * len(self.train_dataset)) // self.hparams.batch_size))
        self.max_iteration = (len(self.train_dataset) * self.hparams.max_epochs) // (self.hparams.batch_size + 1)

        linear = tn_resnet.Linear if self.hparams.use_trunc_norm else torch.nn.Linear
        self.classifier = torch.nn.Sequential(linear(self.hparams.embedding_dim, self.hparams.num_classes))

        # Metrics to log.
        self.train_precision = pl.metrics.Precision(num_classes=self.hparams.num_classes)
        self.val_precision = pl.metrics.Precision(num_classes=self.hparams.num_classes)
        self.train_recall = pl.metrics.Recall(num_classes=self.hparams.num_classes)
        self.val_recall = pl.metrics.Recall(num_classes=self.hparams.num_classes)
        self.train_f1 = pl.metrics.F1(num_classes=self.hparams.num_classes)
        self.val_f1 = pl.metrics.F1(num_classes=self.hparams.num_classes)

    def forward(self, x):
        with torch.no_grad():
            embedding = self.model(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            x = self.train_transform(x)
        y_hat = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        topk_accuracies = utils.calculate_accuracy(output=y_hat, target=y, topk=self.hparams.topk)
        for i, a in enumerate(topk_accuracies):
            self.log(f"train_top-{self.hparams.topk[i]}_accuracy", a, on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        self.log("train_class_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_precision", self.train_precision(y_hat, y), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log("train_recall", self.train_recall(y_hat, y), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log("train_f1", self.train_f1(y_hat, y), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        return loss

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        topk_accuracies = utils.calculate_accuracy(output=y_hat, target=y, topk=self.hparams.topk)
        for i, a in enumerate(topk_accuracies):
            self.log(f"val_top-{self.hparams.topk[i]}_accuracy", a, on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        self.log("val_class_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.val_precision(y_hat, y), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log("val_recall", self.val_recall(y_hat, y), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log("val_f1", self.val_f1(y_hat, y), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.classifier.parameters(), lr=self.hparams.lr,
                                    momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
        )
