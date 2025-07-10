from pathlib import Path

import pytorch_lightning as pl

from models.base.baseline_model import BaselineModel
from config import hparams
from models.base.linear_model import LinearModel
from models.architectures.byol_model import BYOLModel
from models.base.semi_sup_model import SemiSupModel
from models.architectures.byolhp_model import BYOLHPModel
from models.architectures.simclr_model import SimCLRModel
from models.architectures.simclrhp_model import SimCLRHPModel
from models.architectures.vicreg_model import VICRegModel
from models.architectures.vicreghp_model import VICRegHPModel


def run():
    # Seed the PRNG.
    pl.seed_everything(hparams.seed)

    hparams.dhmc_data = "dhmc" in hparams.dataset_name

    # Find the mean and std.
    if hparams.dataset_name == "dhmc_cd":
        hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/CD")
        # Per-channel mean and standard deviation.
        hparams.mean = [0.8655948638916016, 0.8134953379631042, 0.8512848615646362]
        hparams.std = [0.1588381975889206, 0.2300935685634613, 0.18215394020080566]
        # Classes: Abnormal, Normal, Sprue
        hparams.num_classes = 3

    elif hparams.dataset_name == "dhmc_rcc":
        hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/RCC")
        # Per-channel mean and standard deviation.
        hparams.mean = [0.7893273234367371, 0.6810887455940247, 0.7918950915336609]
        hparams.std = [0.19146229326725006, 0.24683408439159393, 0.1701047122478485]
        # Classes: Benign, Chromophobe, Clearcell, Oncocytoma, Papillary
        hparams.num_classes = 5

    elif hparams.dataset_name == "dhmc_luad":
        hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/LUAD")
        # Per-channel mean and standard deviation.
        hparams.mean = [0.8261354565620422, 0.739959180355072, 0.8314031362533569]
        hparams.std = [0.17794468998908997, 0.2484838217496872, 0.15636992454528809]
        # Classes: Acinar, Lepidic, Micropapillary, Papillary, Solid
        hparams.num_classes = 5

    else:
        hparams.data_path = Path("")
        # Assume ImageNet1k
        # Per-channel mean and standard deviation.
        hparams.mean = [0.485, 0.456, 0.406]
        hparams.std = [0.229, 0.224, 0.225]
        # Assume ImageNet1k
        hparams.num_classes = 1000

    # Learning rate scaling.
    hparams.lr = hparams.lr * (hparams.batch_size / 256)

    # Find the mode.
    if hparams.mode == "byol":
        model = BYOLModel(hparams)
    elif "linear" in hparams.mode:
        model = LinearModel(hparams)
    elif hparams.mode == "semi_sup":
        model = SemiSupModel(hparams)
    elif hparams.mode == "baseline":
        model = BaselineModel(hparams)
    elif hparams.mode == "byolhp":
        model = BYOLHPModel(hparams)
    elif hparams.mode == "simclr":
        model = SimCLRModel(hparams)
    elif hparams.mode == "simclrhp":
        model = SimCLRHPModel(hparams)
    elif hparams.mode == "vicreg":
        model = VICRegModel(hparams)
    elif hparams.mode == "vicreghp":
        model = VICRegHPModel(hparams)
    else:
        raise NotImplementedError

    # Set up the trainer.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss_epoch" if (not ("linear" in hparams.mode)) and (hparams.mode != "baseline") and (hparams.mode != "semi_sup") else "val_class_loss_epoch", save_top_k=-1, filename="{epoch}")
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         accumulate_grad_batches=hparams.accumulate_grad_batches,
                         num_sanity_val_steps=hparams.num_sanity_val_steps,
                         checkpoint_callback=checkpoint_callback,
                         log_every_n_steps=hparams.log_every_n_steps,
                         fast_dev_run=hparams.fast_dev_run,
                         default_root_dir=hparams.default_root_dir.joinpath(hparams.mode),
                         deterministic=hparams.deterministic,
                         benchmark=hparams.benchmark,
                         callbacks=[lr_callback])
    # Run the model.
    model.trainer = trainer
    trainer.fit(model)


if __name__ == "__main__":
    run()
