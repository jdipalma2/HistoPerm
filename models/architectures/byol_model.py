import math
from functools import partial

import pytorch_lightning as pl
import torch

from training.loss import mse_loss
from utilities import utils


class BYOLModel(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        # Online models
        # Encoder
        self.online_encoder = utils.get_encoder(self.hparams.encoder_arch)
        # Projector
        self.online_projector = utils.MLP(self.hparams.embedding_dim, self.hparams.mlp_hidden_dim, self.hparams.dim,
                                          num_layers=self.hparams.projection_mlp_layers,
                                          use_trunc_norm=self.hparams.use_trunc_norm,
                                          normalization=partial(torch.nn.BatchNorm1d,
                                                                num_features=self.hparams.mlp_hidden_dim))
        # Predictor
        self.online_predictor = utils.MLP(self.hparams.dim, self.hparams.mlp_hidden_dim, self.hparams.dim,
                                          num_layers=self.hparams.prediction_mlp_layers,
                                          use_trunc_norm=self.hparams.use_trunc_norm,
                                          normalization=partial(torch.nn.BatchNorm1d,
                                                                num_features=self.hparams.mlp_hidden_dim))

        # Target models
        # Encoder
        self.target_encoder = utils.get_encoder(self.hparams.encoder_arch)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # Projector
        self.target_projector = utils.MLP(self.hparams.embedding_dim, self.hparams.mlp_hidden_dim, self.hparams.dim,
                                          num_layers=self.hparams.projection_mlp_layers,
                                          use_trunc_norm=self.hparams.use_trunc_norm,
                                          normalization=partial(torch.nn.BatchNorm1d,
                                                                num_features=self.hparams.mlp_hidden_dim))
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x_1, x_2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        emb_q = self.online_encoder(x_1)
        q_projection = self.online_projector(emb_q)
        q = self.online_predictor(q_projection)  # queries: NxC
        q = torch.nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.target_projector(self.target_encoder(x_2))  # keys: NxC
            k = torch.nn.functional.normalize(k, dim=1)

        return emb_q, q, k

    def forward(self, x):
        return self.online_encoder(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, __ = batch  # batch is a tuple, we just want the image

        with torch.no_grad():
            x_1 = self.view_1_data_transform(x.detach().clone())
            x_2 = self.view_2_data_transform(x.detach().clone())

        emb_q, q, k = self._get_embeddings(x_1, x_2)
        emb_q2, q2, k2 = self._get_embeddings(x_2, x_1)

        loss = (mse_loss(q, k) + mse_loss(q2, k2)).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        with torch.no_grad():
            self._ema_step()

        return loss

    def _get_m(self):
        return 1 - (1 - self.hparams.m) * (math.cos(math.pi * self.global_step / self.max_iteration) + 1) / 2

    @torch.no_grad()
    def _ema_step(self):
        """
        Momentum update of the key encoder
        """
        m = self._get_m()
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
