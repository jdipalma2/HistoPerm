import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from training.loss import covariance_loss, variance_loss
from utilities import utils


class VICRegModel(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        # Models
        # Encoder
        self.encoder = utils.get_encoder(self.hparams.encoder_arch)
        # Projector
        self.projector = utils.Projector(args=self.hparams, embedding=self.hparams.embedding_dim)
        # Find the number of features for computing covariance regularization loss term
        self.num_features = int(self.hparams.mlp.split("-")[-1])

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, __ = batch  # batch is a tuple, we just want the image

        with torch.no_grad():
            x_1 = self.view_1_data_transform(x.detach().clone())
            x_2 = self.view_2_data_transform(x.detach().clone())

        x_1 = self.projector(self.encoder(x_1))
        x_2 = self.projector(self.encoder(x_2))

        # Invariance loss
        inv_loss = F.mse_loss(x_1, x_2)

        # Variance loss
        var_loss = variance_loss(x_1=x_1, x_2=x_2, mach_eps=self.hparams.macheps)

        # Covariance loss
        cov_loss = covariance_loss(x_1=x_1, x_2=x_2, num_features=self.num_features, len_x=len(x))

        self.log("train_variance_loss", var_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_invariance_loss", inv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_covariance_loss", cov_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        loss = (self.hparams.var_coeff * var_loss) + (self.hparams.inv_coeff * inv_loss) + (self.hparams.cov_coeff * cov_loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
