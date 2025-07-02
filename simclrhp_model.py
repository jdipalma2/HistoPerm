import math
from functools import partial

import pytorch_lightning as pl
import torch

from training.loss import info_nce_loss
from utilities import utils


class SimCLRHPModel(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        # Models
        # Encoder
        self.encoder = utils.get_encoder(self.hparams.encoder_arch)
        # Projector
        self.projector = utils.MLP(self.hparams.embedding_dim, self.hparams.mlp_hidden_dim, self.hparams.dim,
                                   num_layers=self.hparams.projection_mlp_layers,
                                   use_trunc_norm=self.hparams.use_trunc_norm,
                                   normalization=partial(torch.nn.BatchNorm1d,
                                                         num_features=self.hparams.mlp_hidden_dim))

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x_in, class_labels = batch  # batch is a tuple, we just want the image

        with torch.no_grad():
            x_1 = self.view_1_data_transform(x_in.detach().clone())
            x_2 = self.view_2_data_transform(x_in.detach().clone())

            sp = math.ceil(self.hparams.shuffle_percentage * len(x_in))

            # Only worry about a subset of the list for shuffling.
            cl = class_labels[:sp].cpu()

            shuf = torch.zeros(self.hparams.num_classes, len(cl), dtype=torch.int64, requires_grad=False)
            # Shuffle the images based on classes.
            for c in range(self.hparams.num_classes):
                class_c = (cl == c).nonzero(as_tuple=True)[0]

                shuf[c][class_c] = class_c[torch.randperm(class_c.numel())]

            shuf = shuf.to(device=x_in.device, non_blocking=True)
            # Shuffle one of the views.
            # Doesn't matter which one.
            # Arbitrary choice to shuffle the first set of views.
            x_1[:sp] = x_1[:sp][torch.max(shuf, dim=0)[0]]

            # Now shuffle everything to avoid learning the first part is weird in terms of views not coming from the same source.
            rand_shuf = torch.randperm(n=len(x_in), device=x_in.device)
            x_1 = x_1[rand_shuf]
            x_2 = x_2[rand_shuf]

        x = torch.cat([x_1, x_2], dim=0)

        features = self.projector(self.encoder(x))
        logits, labels = info_nce_loss(features=features, batch_size=self.hparams.batch_size)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
