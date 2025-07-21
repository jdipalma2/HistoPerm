from functools import partial

import pytorch_lightning as pl
import torch

from training.simclr import info_nce_loss
from utilities import utils
from view_perm.permute_views import PermuteViews


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

        # TODO Find better way to do this
        if self.hparams.use_histoperm:
            self.view_perm = PermuteViews(shuffle_percentage=self.hparams.shuffle_percentage,
                                          num_classes=self.hparams.num_classes,
                                          view_1_data_transform=self.view_1_data_transform,
                                          view_2_data_transform=self.view_2_data_transform)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch  # batch is a tuple, we just want the image

        # TODO Do this in a better way so the conditional isn't re-evaluated at every step
        if self.hparams.use_histoperm:
            x_1, x_2 = self.view_perm.permute_batch(x, y)
        else:
            with torch.no_grad():
                x_1 = self.view_1_data_transform(x.detach().clone())
                x_2 = self.view_2_data_transform(x.detach().clone())

        features = self.projector(self.encoder(torch.cat([x_1, x_2], dim=0)))
        logits, labels = info_nce_loss(features=features, batch_size=self.hparams.batch_size)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
