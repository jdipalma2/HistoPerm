import argparse
from pathlib import Path

# Command line arguments.
parser = argparse.ArgumentParser(description="BYOL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Encoder model
parser.add_argument("--encoder_arch", type=str, default="tn_resnet18", help="Model to use for encoding views.")
parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension for MLP.")

# Data-related
parser.add_argument("--dataset_name", type=str, choices=("dhmc_cd", "dhmc_rcc", "dhmc_luad", "imagenet"))
# Setting dhmc_data = True makes a few assumptions:
#   1. The data is of constant size. If not, this will cause issues with the dataloader.
#   2. The mean and standard deviation values are contained in a file named image_stats.pickle.
parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training.")

# BYOL method
parser.add_argument("--dim", type=int, default=256, help="Middle dimension in projector.")
parser.add_argument("--m", type=float, default=0.97, help="Momentum value for target network.")
parser.add_argument("--exclude_matching_parameters_from_lars", type=list, default=[".bias", ".bn"], help="Parameters to exclude from LARS optimizer.")

# VICReg method
parser.add_argument("--var_coeff", type=float, default=25.0, help="Variance regularization loss coefficient.")
parser.add_argument("--inv_coeff", type=float, default=25.0, help="Invariance regularization loss coefficient.")
parser.add_argument("--cov_coeff", type=float, default=1.0, help="Covariance regularization loss coefficient.")
parser.add_argument("--macheps", type=float, default=0.0001, help="Epsilon value for variance regularization loss term.")
parser.add_argument("--mlp", type=str, default="2048-2048", help="Size and number of layers in the MLP expander.")

# Optimizer
parser.add_argument("--lr", type=float, default=0.45, help="Learning rate for optimization.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimization.")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimization.")
parser.add_argument("--max_epochs", type=int, default=50, help="Number of training epochs.")

# Transform
parser.add_argument("--crop_size", type=int, default=224, help="Size to crop input data to.")

# MLP
parser.add_argument("--projection_mlp_layers", type=int, default=2, help="Number of layers in projector.")
parser.add_argument("--prediction_mlp_layers", type=int, default=2, help="Number of layers in predictor.")
parser.add_argument("--mlp_hidden_dim", type=int, default=4096, help="Number of hidden neurons in MLP.")

parser.add_argument("--prediction_mlp_normalization", type=str, default="same", help="Same normalization technique for predictor.")

# Data loading
parser.add_argument("--num_data_workers", type=int, default=16, help="Number of subprocesses to use for data loading.")
parser.add_argument("--drop_last_batch", type=bool, default=False, help="Whether to drop the last mini-batch.")
parser.add_argument("--pin_data_memory", type=bool, default=True, help="Pin memory in GPU.")


# Gradient accumulation
parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of gradient steps to accumulate before optimizer step.")

# CUDA
parser.add_argument("--deterministic", type=bool, default=False, help="Use deterministic CUDA algorithms.")
parser.add_argument("--benchmark", type=bool, default=True, help="Benchmark CUDA algorithms for speed.")

# RNG
parser.add_argument("--seed", type=int, default=42, help="Seed for PRNG.")

# View 1
parser.add_argument("--view_1_crop_prob", type=float, default=1.0, help="Probability of applying random resized cropping for view 1.")
parser.add_argument("--view_1_hor_flip_prob", type=float, default=0.5, help="Probability of applying random horizontal flipping for view 1.")
parser.add_argument("--view_1_vert_flip_prob", type=float, default=0.5, help="Probability of applying random vertical flipping for view 1.")
parser.add_argument("--view_1_cj_brightness", type=float, default=0.4, help="Color jitter brightness for view 1.")
parser.add_argument("--view_1_cj_contrast", type=float, default=0.4, help="Color jitter contrast for view 1.")
parser.add_argument("--view_1_cj_hue", type=float, default=0.1, help="Color jitter hue for view 1.")
parser.add_argument("--view_1_cj_saturation", type=float, default=0.2, help="Color jitter saturation for view 1.")
parser.add_argument("--view_1_cj_prob", type=float, default=0.8, help="Probability of applying color jitter transform for view 1.")
parser.add_argument("--view_1_gs_prob", type=float, default=0.2, help="Probability of converting view 1 to grayscale.")
parser.add_argument("--view_1_gauss_sigma", type=tuple, default=(0.1, 2.0), help="Range to select sigma from for gaussian filter.")
parser.add_argument("--view_1_gauss_blur_divider", type=int, default=10, help="Scalar to determine size of gaussian blur kernel size.")
parser.add_argument("--view_1_gauss_prob", type=float, default=1.0, help="Probability of applying gaussian blur to view 1.")
parser.add_argument("--view_1_solarize_threshold", type=int, default=128, help="Solarization threshold for view 1.")
parser.add_argument("--view_1_solarize_prob", type=float, default=0.0, help="Probability of applying solarization to view 1.")

# View 2
parser.add_argument("--view_2_crop_prob", type=float, default=1.0, help="Probability of applying random resized cropping for view 2.")
parser.add_argument("--view_2_hor_flip_prob", type=float, default=0.5, help="Probability of applying random horizontal flipping for view 2.")
parser.add_argument("--view_2_vert_flip_prob", type=float, default=0.5, help="Probability of applying random vertical flipping for view 2.")
parser.add_argument("--view_2_cj_brightness", type=float, default=0.4, help="Color jitter brightness for view 2.")
parser.add_argument("--view_2_cj_contrast", type=float, default=0.4, help="Color jitter contrast for view 2.")
parser.add_argument("--view_2_cj_hue", type=float, default=0.1, help="Color jitter hue for view 2.")
parser.add_argument("--view_2_cj_saturation", type=float, default=0.2, help="Color jitter saturation for view 2.")
parser.add_argument("--view_2_cj_prob", type=float, default=0.8, help="Probability of applying color jitter transform for view 2.")
parser.add_argument("--view_2_gs_prob", type=float, default=0.2, help="Probability of converting view 2 to grayscale.")
parser.add_argument("--view_2_gauss_sigma", type=tuple, default=(0.1, 2.0), help="Range to select sigma from for gaussian filter.")
parser.add_argument("--view_2_gauss_blur_divider", type=int, default=10, help="Scalar to determine size of gaussian blur kernel size.")
parser.add_argument("--view_2_gauss_prob", type=float, default=0.1, help="Probability of applying gaussian blur to view 2.")
parser.add_argument("--view_2_solarize_threshold", type=int, default=128, help="Solarization threshold for view 2.")
parser.add_argument("--view_2_solarize_prob", type=float, default=0.2, help="Probability of applying solarization to view 2.")

# Mode
parser.add_argument("--mode", type=str, default="pretrain", help="BYOL mode.")

# PyTorch Lightning
parser.add_argument("--default_root_dir", type=Path, default=Path("logs"), help="Where to store logs.")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of validation steps to ensure model runs.")
parser.add_argument("--fast_dev_run", type=bool, default=False, help="")
parser.add_argument("--log_every_n_steps", type=int, default=1, help="Number of steps between saving logs.")

# Linear and semi-supervised mode specific
parser.add_argument("--ckpt_file", type=Path, default=Path(""), help="Checkpoint file to use as either frozen encoder or weight initialization.")

# Accuracy
parser.add_argument("--topk", type=tuple, default=(1, 2), help="The top-k accuracy values to report.")

# Optimizer
parser.add_argument("--eta", type=float, default=1e-3, help="Eta value for LARS optimizer.")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for linear warmup learning rate scaling.")

parser.add_argument("--use_trunc_norm", type=bool, default=True, help="Truncated normal initializer for parameters.")

parser.add_argument("--shuffle_percentage", type=float, default=1, help="Percentage of batch to shuffle.")

hparams = parser.parse_args()
