"""Command-line parser of src/train.py (moved out of train.py, audit W10, 2026-11-14).

A pure move: `python src/train.py --help` is byte-identical before and after. Post-parse defaults
and checks (depth mode, selection metric, symmetric loss, bbox range, DDP) stay in train.py.
"""

import argparse

from src.bop_dataset import PIXEL_AUGS
from src.models import OBJECT_CONDITIONINGS, SUPPORTED_VIT_MODELS
from src.utils import DEPTH_MODES

BOP_DATASETS = ("LMO", "YCBV", "TLESS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FastPoseViT on SPEED dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SPEED", "SPEED_PLUS_SYNTHETIC", "SPEED_PLUS", *BOP_DATASETS],
        default="SPEED",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory of the dataset (auto-determined from dataset choice if not specified)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the best model",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Directory to store TensorBoard logs"
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        choices=SUPPORTED_VIT_MODELS,
        default="vit_b_16",
        help="ViT model to use",
    )
    parser.add_argument(
        "--vit_weights", type=str, default=None, help="Path to pretrained ViT weights"
    )
    parser.add_argument(
        "--model_weights", type=str, default=None, help="Path to pretrained full model weights"
    )

    # Model parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and validation"
    )
    parser.add_argument(
        "--max_lr", type=float, default=1e-4, help="Maximum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "muon"],
        default="adam",
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[
            "cosineannealinglr",
            "cosineannealingwarmrestarts",
            "logarithmiclr",
        ],
        default="cosineannealinglr",
        help="Learning rate scheduler to use (ignored for muon optimizer)",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=5,
        help="Number of hidden layers in MLPWithProjection",
    )
    parser.add_argument(
        "--hidden_layer_dim",
        type=int,
        default=512,
        help="Dimension of hidden layers in MLPWithProjection",
    )
    parser.add_argument(
        "--translation_loss",
        type=str,
        choices=[
            "mse",
            "mae",
            "frobenius",
            "relative_mse",
            "relative_frobenius",
            "lp_norm_075",
            "lp_norm_3",
            "lp_norm_inf",
            "nuclear",
            "relative_translation",
            "decoupled",
        ],
        default="mse",
        help="Loss function for translation",
    )
    parser.add_argument(
        "--decoupled_lambda",
        type=float,
        default=1.0,
        help="Weight of the U_z term in --translation_loss decoupled",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="quaternion",
        help="Format for representing rotations",
    )
    parser.add_argument(
        "--rotation_loss",
        type=str,
        choices=[
            "mse",
            "geodesic",
            "simplified",
            "frobenius",
            "mae",
            "lp_norm_075",
            "lp_norm_3",
            "lp_norm_inf",
            "nuclear",
            "6d",
            "6d_mae",
            "6d_frobenius",
            "6d_mse",
            "6d_lp_norm_075",
            "6d_lp_norm_3",
            "6d_lp_norm_inf",
            "6d_nuclear",
            "6d_relative_frobenius",
            "relative_frobenius",
        ],
        default="mse",
        help="Loss function for rotation",
    )
    parser.add_argument(
        "--nb_class_tokens",
        type=int,
        default=1,
        help="Number of class tokens to use in the model",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Use layer normalization in MLP layers",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for MLP layers (0.0 = no dropout)",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Use residual connections in MLP layers",
    )

    # Experiments arguments
    # Don't use pixel data augmentation
    parser.add_argument(
        "--no_pixel_augmentation",
        action="store_true",
        help="Don't use pixel data augmentation",
    )
    # Don't use spatial data augmentation
    parser.add_argument(
        "--no_spatial_augmentation",
        action="store_true",
        help="Don't use spatial data augmentation",
    )
    # Use domain gap pixel augmentation
    parser.add_argument(
        "--domain_gap_pixel_augmentation",
        action="store_true",
        help="Use domain gap pixel augmentation with sun flare, blur, noise, compression, and dropout",
    )
    # Use style augmentation
    parser.add_argument(
        "--do_style_aug",
        action="store_true",
        help="Enable neural style augmentation (requires --domain_gap_pixel_augmentation)",
    )
    # Use absolute translations instead of relative translations
    parser.add_argument(
        "--use_absolute_translation",
        action="store_true",
        help="Use absolute translations instead of relative translations for ablation study",
    )
    # Use full image instead of cropped image
    parser.add_argument(
        "--use_full_image",
        action="store_true",
        help="Use full image instead of cropped image for ablation study",
    )
    # Don't use rotation-compensating correction factor in loss
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="Don't use rotation-compensating correction factor in loss",
    )
    # Don't use bbox augmentation
    parser.add_argument(
        "--no_bbox_augmentation",
        action="store_true",
        help="Don't use bbox augmentation during spatial augmentation",
    )
    # Bbox augmentation crop percentage
    parser.add_argument(
        "--bbox_crop_percent",
        type=float,
        default=10.0,
        help="Maximum crop percentage for bbox augmentation (0-50 percent, default: 10.0)",
    )
    # Don't pad cropped images to make them square
    parser.add_argument(
        "--no_crop_padding",
        action="store_true",
        help="Don't pad cropped images to make them square",
    )
    # Merge heads
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        help="Merge translation and rotation outputs into a single MLP",
    )
    # Skip MLP layers
    parser.add_argument(
        "--no_mlp",
        action="store_true",
        help="Skip MLP layers and use direct projection only",
    )
    # Skip tensorboard logging
    parser.add_argument(
        "--skip_tensorboard",
        action="store_true",
        help="Skip tensorboard logging to save time and reduce I/O",
    )
    # Convert images to grayscale
    parser.add_argument(
        "--convert_to_grayscale",
        action="store_true",
        help="Convert images to grayscale (replicated as 3-channel for model compatibility)",
    )

    # Training optimization arguments
    parser.add_argument(
        "--no_compile_model",
        action="store_true",
        help="Disable torch.compile optimization (enabled by default)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Maximum gradient norm for gradient clipping (0.0 disables clipping)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training: 'auto' (from --checkpoint_path if it exists, else start fresh) "
        "or the path of a checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Where to save the checkpoint for resuming, every epoch (default: LOG_DIR/last.pt)",
    )
    parser.add_argument(
        "--use_per_epoch_scheduling",
        action="store_true",
        help="Use per-epoch learning rate scheduling instead of per-iteration (more conservative, may help convergence)",
    )

    # BOP datasets (src/bop_dataset.py; data from $BOP_PATH or $NAS_HOME/datasets/bop)
    parser.add_argument(
        "--depth_mode",
        type=str,
        choices=DEPTH_MODES,
        default=None,
        help="Depth parameterisation of U_z (default: site for BOP, legacy for SPEED)",
    )
    parser.add_argument(
        "--bop_obj_ids",
        type=int,
        nargs="+",
        default=None,
        help="BOP object ids to train/validate on (default: all objects of the dataset)",
    )
    parser.add_argument(
        "--bop_train_split",
        type=str,
        default="train",
        choices=["train", "train_mix"],
        help="Training split; train_mix = PBR + real images (YCBV: train_real, M11 B3; LMO: the "
        "real LM images of the 8 objects without scene 2 and the BOP19 val images, M17)",
    )
    parser.add_argument(
        "--bop_real_fraction",
        type=float,
        default=None,
        help="M17, with train_mix and --bop_epoch_samples: fraction of each object's samples drawn "
        "from the real images (default: as the rows fall, e.g. 43 %% real on YCB-V, ~3 %% on LM-O)",
    )
    parser.add_argument(
        "--bop_val_split",
        type=str,
        default="val",
        choices=["val", "val_pbr", "val_real"],
        help="Validation split (never the test split; YCBV: val = val_pbr, val_real = held-out train_real; TLESS val_real = held-out train_primesense rings)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Validate on a fixed random subset of this size (BOP)",
    )
    parser.add_argument(
        "--bop_pixel_aug",
        type=str,
        default="standard",
        choices=[a for a in PIXEL_AUGS if a != "none"],
        help="BOP pixel augmentation (off with --no_pixel_augmentation)",
    )
    parser.add_argument(
        "--bop_bbox_source",
        type=str,
        default="visib",
        choices=["visib", "obj", "mix"],
        help="BOP: box the training crops use: bbox_visib, bbox_obj (amodal), or a random mix "
        "per sample (M7). Validation always uses bbox_visib",
    )
    parser.add_argument(
        "--crop_context",
        action="store_true",
        help="BOP: fill the square crop around the bbox with image instead of black",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=1.0,
        help="BOP (M11): side of the square crop in units of max(w, h) of the box (1 = tight); "
        "use with --crop_context so that the margin shows the image",
    )
    parser.add_argument(
        "--bop_occlusion_aug",
        action="store_true",
        help="BOP (M11): erase 1-4 random rectangles of each training crop (p = 0.5)",
    )
    parser.add_argument(
        "--bop_occlusion_p",
        type=float,
        default=0.5,
        help="BOP (M18.7): probability of the --bop_occlusion_aug rectangles",
    )
    parser.add_argument(
        "--bop_occlusion_max_holes",
        type=int,
        default=4,
        help="BOP (M18.7): largest number of --bop_occlusion_aug rectangles",
    )
    parser.add_argument(
        "--bop_occlusion_max_size",
        type=float,
        default=0.3,
        help="BOP (M18.7): largest side of those rectangles, as a fraction of the crop side",
    )
    parser.add_argument(
        "--bop_cutout_p",
        type=float,
        default=0.0,
        help="BOP (M18.7): probability of pasting the visible pixels of another object over a "
        "training crop (its pixels leave the dense mask / XYZ targets); 0 = off",
    )
    parser.add_argument(
        "--bop_real_background",
        action="store_true",
        help="BOP (M17, train_mix): paste the object of each real training image onto a random PBR "
        "image with its mask_visib (T-LESS train_primesense has black backgrounds)",
    )
    parser.add_argument(
        "--translation_loss_weight",
        type=float,
        default=1.0,
        help="M11: total loss = weight * translation loss + rotation weight * rotation loss",
    )
    parser.add_argument(
        "--rotation_loss_weight",
        type=float,
        default=1.0,
        help="M18.7: weight of the rotation loss in the total training loss",
    )
    parser.add_argument(
        "--dense_heads",
        action="store_true",
        help="BOP (M18): add the decoder on the patch tokens (visible mask + normalised object "
        "coordinates) and train it with the pose losses; needs a ViT backbone",
    )
    parser.add_argument(
        "--dense_size", type=int, default=64, help="M18: side of the dense outputs / targets"
    )
    parser.add_argument(
        "--dense_symmetric_xyz",
        action="store_true",
        help="M18.6a: the XYZ target of a symmetric object is the one of the symmetric pose the "
        "rotation loss picks for the predicted rotation (needs --dense_heads and the symmetric "
        "rotation loss); off = the GT pose as is",
    )
    parser.add_argument(
        "--mask_loss_weight", type=float, default=1.0, help="M18: weight of the mask BCE"
    )
    parser.add_argument(
        "--xyz_loss_weight", type=float, default=1.0, help="M18: weight of the XYZ L1"
    )
    parser.add_argument(
        "--bop_overfit",
        type=int,
        default=0,
        help="BOP sanity check: train on this many fixed instances, no augmentation, val on the same",
    )
    parser.add_argument("--bop_overfit_seed", type=int, default=0)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=None,
        choices=["val_loss", "bop_score"],
        help="Best model (model.pth): lowest val loss, or highest (AR_MSSD + AR_MSPD) / 2 "
        "(default: bop_score for BOP, val_loss for SPEED)",
    )
    parser.add_argument(
        "--symmetric_rotation_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="BOP: min of the 6D relative Frobenius loss over each object's symmetries (M5; "
        "default on for BOP with --rotation_format matrix and 6d_relative_frobenius, else off)",
    )
    parser.add_argument(
        "--object_conditioning",
        type=str,
        default="heads",
        choices=OBJECT_CONDITIONINGS,
        help="BOP multi-object model: per-object heads (a), per-object class token (b) or both",
    )
    parser.add_argument(
        "--bop_epoch_samples",
        type=int,
        default=0,
        help="BOP: virtual epochs of this many samples, the same number per object "
        "(default 0: one shuffled pass over all training rows)",
    )
    parser.add_argument(
        "--bop_sampler_seed",
        type=int,
        default=0,
        help="Seed of the --bop_epoch_samples order (the order depends only on seed and epoch)",
    )
    return parser
