import os
import torch
import random
import argparse
import numpy as np

from src.train_action import train_har
from src.train_privacy import train_pri


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()

# Job parameters
parser.add_argument('--run_id', type=str, default='debug', help='Job run identifier')
parser.add_argument('--task', type=str, default='har', help='Task', choices=['har', 'pri'])
parser.add_argument('--saved_model', type=str, default=None, help='Path to saved model weights')
parser.add_argument('--ckpt_model', type=str, default=None, help='Path to ckpt model weights')
parser.add_argument('--hide', action='store_true', help='Hide secret or not')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--devices', default=[0], help='GPU id')

# Dataset parameters
parser.add_argument('--train_data', type=str, default='ucf101', help='Train dataset')
parser.add_argument('--val_data', type=str, default='ucf101', help='Val dataset')
parser.add_argument('--reso_h', type=int, default=224, help='Resized frame height')
parser.add_argument('--reso_w', type=int, default=224, help='Resized frame width')
parser.add_argument('--ori_reso_h', type=int, default=240, help='Original frame height')
parser.add_argument('--ori_reso_w', type=int, default=320, help='Original frame width')
parser.add_argument('--num_classes', type=int, default=101, help='Number of classes')
parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per input clip')
parser.add_argument('--fix_skip', type=int, default=2, help='Frame skipping rate during data loading')
parser.add_argument('--num_modes', type=int, default=5, help='Number of temporal augmentation modes')
parser.add_argument('--num_skips', type=int, default=1, help='Number of skip strategies')
parser.add_argument('--data_percentage', type=float, default=1.0, help='Percentage of data to use (0.0-1.0)')

# Training parameters
parser.add_argument('--model', type=str, default='r3dpro_ta', help='Model to use',
                    choices=['r3d18', '4r3d', 'r3dpro', 'r3dpro_ta', 'r50', 'vit'])
parser.add_argument('--hide_model', type=str, default='lfvsn', help='HideModel to use',
                    choices=['lfvsn', 'hinet', 'wengnet', 'hidden'])
parser.add_argument('--alpha', type=float, default=0.0, help='Parameter of spatial promotion')
parser.add_argument('--beta', type=float, default=0.0, help='Parameter of temporal promotion')
parser.add_argument('--theta', type=float, default=0.2, help='Parameter of CBDA')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--cov_ran', action='store_true', help='Random select cover')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
parser.add_argument('--pin_memory', action='store_true', help='Dataloader pin memory')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate')
parser.add_argument('--num_epochs', type=int, default=200, help='Total training epochs')
parser.add_argument('--val_int', type=int, default=10, help='Interval of val')
parser.add_argument('--opt_type', type=str, default='adam', help='Optimizer type (adam/sgd/etc.)')
parser.add_argument('--lr_scheduler', type=str, default='loss_based', help='Learning rate scheduler type')
parser.add_argument('--warmup', action='store_true', help='Enable learning rate warmup')
parser.add_argument('--warmup_array', default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], help='Enable learning rate warmup')
parser.add_argument('--betas', nargs=2, type=float, default=(0.9, 0.999), help='Adam optimizer beta parameters')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coefficient')

# Validation augmentation
parser.add_argument('--hflip', nargs='+', type=int, default=[0],
                    help='Horizontal flip probabilities (e.g., [0] = no flip)')
parser.add_argument('--cropping_facs', nargs='+', type=float, default=[0.8],
                    help='Cropping factors for validation (e.g., [0.8])')
parser.add_argument('--weak_aug', action='store_true', help='Use weaker augmentation for validation')
parser.add_argument('--no_ar_distortion', action='store_true', help='Disable aspect ratio distortion in validation')
parser.add_argument('--aspect_ratio_aug', action='store_true', help='Enable aspect ratio augmentation in validation')

args = parser.parse_args()


def main(args):
    if args.task == 'har':
        train_har(args)
    elif args.task == 'pri':
        train_pri(args)


if __name__ == '__main__':
    seed_torch(args.seed)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    args.devices = list(range(torch.cuda.device_count()))
    main(args)
