from functools import partial

import pytorch_lightning as pl
import torch

from training.loss import info_nce_loss
from utilities import utils


class SimCLRModel(pl.LightningModule):

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
        x, __ = batch  # batch is a tuple, we just want the image

        with torch.no_grad():
            x = torch.cat(
                [self.view_1_data_transform(x.detach().clone()), self.view_2_data_transform(x.detach().clone())], dim=0)

        features = self.projector(self.encoder(x))
        logits, labels = info_nce_loss(features=features, batch_size=self.hparams.batch_size)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
