import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from training.vicreg import covariance_loss, variance_loss
from utilities import utils
from view_perm.permute_views import PermuteViews


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

        # TODO Find better way to do this
        if self.hparams.use_histoperm:
            self.view_perm = PermuteViews(shuffle_percentage=self.hparams.shuffle_percentage,
                                          num_classes=self.hparams.num_classes,
                                          view_1_data_transform=self.view_1_data_transform,
                                          view_2_data_transform=self.view_2_data_transform)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch

        # TODO Do this in a better way so the conditional isn't re-evaluated at every step
        if self.hparams.use_histoperm:
            x_1, x_2 = self.view_perm.permute_batch(x, y)
        else:
            with torch.no_grad():
                x_1 = self.view_1_data_transform(x.detach().clone())
                x_2 = self.view_2_data_transform(x.detach().clone())

        x_1 = self.projector(self.encoder(x_1))
        x_2 = self.projector(self.encoder(x_2))

        # Invariance loss
        inv_loss = F.mse_loss(x_1, x_2)

        # Variance loss
        var_loss = variance_loss(x_1=x_1, mach_eps=self.hparams.macheps)

        # Covariance loss
        cov_loss = covariance_loss(x_1=x_1, x_2=x_2, num_features=self.num_features, len_x=len(x))

        self.log("train_variance_loss", var_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_invariance_loss", inv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_covariance_loss", cov_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        loss = (self.hparams.var_coeff * var_loss) + (self.hparams.inv_coeff * inv_loss) + (
                self.hparams.cov_coeff * cov_loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
