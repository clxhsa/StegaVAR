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

"""

nohide:
python main.py --run_id r3d_ta_112 --task har --model r3d_ta --batch_size 32 --num_workers 8 --pin_memory --reso_h 112 --reso_w 112 | tee ucf_logs/r3d_ta_112.log
python main.py --run_id nohide_r3d_time --task har --model r3dpro_time --batch_size 32 --num_workers 8 --pin_memory | tee ucf_logs/nohide_r3dpro_time.log
python main.py --run_id nohide_r3d_time_ta --task har --model r3dpro_time_ta --batch_size 32 --num_workers 8 --pin_memory --learning_rate 1e-4 --alpha 0.2 --beta 0.3 --theta 0.3 | tee ucf_logs/nohide_r3dpro_time_ta.log

python main.py --run_id hmdb51_nohide_r3d_time_ta --task har --train_data hmdb51 --val_data hmdb51 --model r3dpro_time_ta --batch_size 32 --num_workers 8 --pin_memory --learning_rate 1e-4 --alpha 0.2 --beta 0.3 --theta 0.0 | tee hmdb_logs/nohide_r3dpro_time_ta.log



hide_har:
python main.py --hide --run_id ucf101_4r3d --task har --model 4r3d --batch_size 64 --num_workers 8 --pin_memory | tee ucf_logs/4r3d.log

python main.py --hide --run_id ucf101_r3dpro --task har --model r3dpro --batch_size 64 --num_workers 8 --pin_memory | tee ucf_logs/r3dpro.log
python main.py --hide --run_id ucf101_r3dpro_03 --task har --model r3dpro --batch_size 64 --num_workers 8 --pin_memory --alpha 0.3 | tee ucf_logs/r3dpro_03.log
python main.py --hide --run_id ucf101_r3dpro --task har --model r3dpro --batch_size 64 --num_workers 8 --pin_memory | tee ucf_logs/r3dpro.log


python main.py --hide --run_id ucf101_r3dpro_time --task har --model r3dpro_time --batch_size 64 --num_workers 8 --pin_memory | tee ucf_logs/r3dpro_time.log
python main.py --hide --run_id ucf101_r3dpro_time_ta --task har --model r3dpro_time_ta --batch_size 64 --num_workers 8 --pin_memory --learning_rate 1e-4 | tee ucf_logs/r3dpro_time_ta.log
python main.py --hide --run_id ucf101_r3dpro_ta --task har --model r3dpro_ta --batch_size 64 --num_workers 8 --pin_memory --learning_rate 1e-4 | tee ucf_logs/r3dpro_ta.log

python main.py --hide --run_id ucf101_r3dpro_time_ta_220 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.2 --theta 0.0 | tee ucf_logs/all_220.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_221 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.2 --theta 0.1 | tee ucf_logs/all_221.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_222 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.2 --theta 0.2 | tee ucf_logs/all_222.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_223 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.2 --theta 0.3 | tee ucf_logs/all_223.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_224 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.2 --theta 0.4 | tee ucf_logs/all_224.log

python main.py --hide --run_id ucf101_r3dpro_time_ta_241 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.4 --theta 0.1 | tee ucf_logs/all_241.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_232 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 | tee ucf_logs/all_232.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_233 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.3 | tee ucf_logs/all_233.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_212 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.1 --theta 0.2 | tee ucf_logs/all_212.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_242 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.4 --theta 0.2 | tee ucf_logs/all_242.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_252 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.5 --theta 0.2 | tee ucf_logs/all_252.log

python main.py --hide --run_id ucf101_r3dpro_time_ta_032 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.0 --beta 0.3 --theta 0.2 | tee ucf_logs/all_032.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_132 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.1 --beta 0.3 --theta 0.2 | tee ucf_logs/all_132.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_322 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.3 --beta 0.2 --theta 0.2 | tee ucf_logs/all_322.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_332 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.3 --beta 0.3 --theta 0.2 | tee ucf_logs/all_332.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_421 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.4 --beta 0.2 --theta 0.1 | tee ucf_logs/all_421.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_532 --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.5 --beta 0.3 --theta 0.2 | tee ucf_logs/all_532.log

python main.py --hide --run_id ucf101_r3dpro_time_ta_232_hinet --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 --hide_model hinet | tee ucf_logs/all_232_hinet.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_232_wengnet --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 --hide_model wengnet | tee ucf_logs/all_232_wengnet.log
python main.py --hide --run_id ucf101_r3dpro_time_ta_232_hidden --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 --hide_model hidden | tee ucf_logs/all_232_hidden.log

python main.py --hide --train_data hmdb51 --val_data hmdb51 --run_id hmdb51_r3dpro_time_ta --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 --saved_model ckpt/r3dpro_time_ta/ucf101_r3dpro_time_ta_232/model_148_bestAcc_0.7131.pth | tee hmdb_logs/all.log
python main.py --hide --train_data hmdb51 --val_data hmdb51 --run_id hmdb51_r3dpro_time_ta_hinet --task har --model r3dpro_time_ta --batch_size 32 --pin_memory --alpha 0.2 --beta 0.3 --theta 0.2 --saved_model ckpt/r3dpro_time_ta/ucf101_r3dpro_time_ta_232_hinet/model_75_bestAcc_0.7007.pth --hide_model hinet | tee hmdb_logs/all_hinet.log

hide_pri:
python main.py --hide --run_id ucf101_r50 --task pri --model r50 --batch_size 64 --num_workers 4 --pin_memory --num_epochs 100 --train_data ucf101 --val_data ucf101 --val_int 1 | tee ucf_logs/r50_pri.log
python main.py --hide --run_id hmdb51_r50 --task pri --model r50 --batch_size 64 --num_workers 4 --pin_memory --num_epochs 100 --train_data hmdb51 --val_data hmdb51 --val_int 1 | tee hmdb_logs/r50_pri.log

python main.py --hide --run_id vispr1_r50 --task pri --model r50 --batch_size 64 --num_workers 4 --pin_memory --num_epochs 100 --train_data vispr1 --val_data vispr1 --val_int 1 | tee vispr_logs/r50_vis1.log
python main.py --hide --run_id vispr1_r50 --task pri --model r50 --batch_size 64 --num_workers 4 --pin_memory --num_epochs 100 --train_data vispr2 --val_data vispr2 --val_int 1 --saved_model ckpt/r50/vispr1_r50_hinet/model_84_bestAP_0.5351.pth | tee vispr_logs/r50_vis2_hinet.log

python main.py --hide --run_id ucf101_r50_tran --task pri --model r50 --batch_size 64 --num_workers 4 --pin_memory --num_epochs 100 --train_data ucf101 --val_data ucf101 --val_int 1 --saved_model r50/hmdb51_r50/model_49_bestAP_0.6241.pth | tee ucf_logs/r50_pri_tran.log

"""
