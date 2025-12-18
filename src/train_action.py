import argparse
import importlib
import numpy as np
import os
# from tensorboardX import SummaryWriter
import time
import random
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from hide.LF_VSN.models.modules import common
from hide.LF_VSN.models.networks import define_G_v2
import hide.LF_VSN.options.options as option
from hide.HiNet.model import *
from src.hide_vid import hide_vid
from hide.wengnet.wengnet import wengnet
from hide.hidden.hidden import HiDDeN


from src.model.get_model import get_har_model, get_pri_model, load_network
from src.dataloader.get_data import *
# from src.dataloader.ucf101 import *

import sys

sys.path.insert(0, '..')

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, data_loader, models, criterion, optimizer,lr, scaler, device_name, params):
    print(f'Train at epoch {epoch}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f'Learning rate is: {param_group["lr"]}')

    cls_losses = []
    mse_3d_losses = []
    mse_time_losses = []
    predictions, gt = [], []

    ft_model = models[0]
    if params.hide:
        fh_model = models[1]
        dwt = models[2]
        iwt = models[3]
    # Set ft model to train.
    ft_model.train()

    ce = criterion[0]
    if 'pro' in params.model:
        mse = criterion[1]

    for i, (secret, label, cover, _) in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)
        secret = secret.to(device=torch.device(device_name), non_blocking=True)
        label = torch.from_numpy(np.asarray(label)).to(device=torch.device(device_name), non_blocking=True)
        if params.hide:
            cover = cover.to(device=torch.device(device_name), non_blocking=True)
            
            stego = hide_vid(params, secret, cover, fh_model, dwt, iwt)
            stego = stego.to(device=torch.device(device_name), non_blocking=True)
            input = stego
        else:
            input = secret


        # Autocast automatic mixed precision.
        with autocast():
            if 'pro' in params.model and (params.alpha != 0.0 or params.beta != 0.0):
                out_map_3d, out_map_time, output = ft_model(
                    stego=input.permute(0, 2, 1, 3, 4),
                    secret=secret.permute(0, 2, 1, 3, 4),
                    is_train=True)

                loss_cls = ce(output, label)

                if params.alpha != 0.0:
                    loss_3d_1 = mse(out_map_3d[0][0], out_map_3d[0][1])
                    loss_3d_2 = mse(out_map_3d[1][0], out_map_3d[1][1])
                    loss_3d_3 = mse(out_map_3d[2][0], out_map_3d[2][1])
                    loss_3d_4 = mse(out_map_3d[3][0], out_map_3d[3][1])
                    mse_3d = loss_3d_1 + loss_3d_2 + loss_3d_3 + loss_3d_4
                else:
                    mse_3d = None

                if params.beta != 0.0:
                    loss_time_1 = mse(out_map_time[0][0], out_map_time[0][1])
                    loss_time_2 = mse(out_map_time[1][0], out_map_time[1][1])
                    loss_time_3 = mse(out_map_time[2][0], out_map_time[2][1])
                    loss_time_4 = mse(out_map_time[3][0], out_map_time[3][1])
                    mse_time = loss_time_1 + loss_time_2 + loss_time_3 + loss_time_4
                else:
                    mse_time = None

                loss = loss_cls
                if mse_3d is not None:
                    loss = loss + params.alpha * mse_3d
                if mse_time is not None:
                    loss = loss + params.beta * mse_time

            else:
                output = ft_model(input.permute(0, 2, 1, 3, 4))
                loss_cls = ce(output, label)
                mse_3d, mse_time = None, None

                loss = loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predictions.extend(torch.max(output, axis=1).indices.cpu().numpy())
            gt.extend(label.cpu().numpy())

        cls_losses.append(loss_cls.item())
        if mse_3d is not None:
            mse_3d_losses.append(mse_3d.item())
        if mse_time is not None:
            mse_time_losses.append(mse_time.item())

        if i % 100 == 0:
            msg = f'Training Epoch {epoch}, Batch {i}, CE: {np.mean(cls_losses):.5f}'
            if len(mse_3d_losses) > 0:
                msg += f', MSE3d: {np.mean(mse_3d_losses):.5f}'
            if len(mse_time_losses) > 0:
                msg += f', MSEtime: {np.mean(mse_time_losses):.5f}'
            print(msg, flush=True)

    msg = f'Training Epoch: {epoch}, CE: {np.mean(cls_losses)}'
    if len(mse_3d_losses) > 0:
        msg += f', MSE3d: {np.mean(mse_3d_losses)}'
    if len(mse_time_losses) > 0:
        msg += f', MSEtime: {np.mean(mse_time_losses)}'
    print(msg)

    predictions = np.asarray(predictions)
    gt = np.asarray(gt)

    accuracy = ((predictions == gt).sum()) / np.size(predictions)
    print(f'Training Accuracy at Epoch {epoch} is {accuracy * 100:0.3f}%')

    del loss, input, cover, output, label

    return ft_model, np.mean(cls_losses), scaler


# Validation epoch.
def val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict, data_loader, models, criterion, device_name,
              params):
    print(f'Validation at epoch {epoch}.')

    ft_model = models[0]
    if params.hide:
        fh_model = models[1]
        dwt = models[2]
        iwt = models[3]
    # Set model to eval.
    ft_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []


    for i, (secret, label, cover , vid_path, _) in enumerate(data_loader):
        if params.hide:
            secret = secret.to(device=torch.device(device_name), non_blocking=True)
            cover = cover.to(device=torch.device(device_name), non_blocking=True)
            stego = hide_vid(params, secret, cover, fh_model, dwt, iwt)
            stego = stego
            input = stego
        else:
            input = secret

        vid_paths.extend(vid_path)
        ground_truth.extend(label)

        input = input.to(device=torch.device(device_name), non_blocking=True)
        label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).to(device=torch.device(device_name),
                                                                                  non_blocking=True)

        with torch.no_grad():
            if 'pro' in params.model:
                output = ft_model(stego=input.permute(0, 2, 1, 3, 4))
            else:
                output = ft_model(input.permute(0, 2, 1, 3, 4))

            loss = criterion[0](output, label)

        losses.append(loss.item())

        predictions.extend(nn.functional.softmax(output, dim=1).cpu().data.numpy())

        if i % 100 == 0:
            print(f'Validation Epoch {epoch}, Batch {i}, Loss : {np.mean(losses)}', flush=True)

    del loss, input, secret, output, label

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
    c_pred = pred_array[:, 0]

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])] = ground_truth[entry]

    correct_count = np.sum(c_pred == ground_truth)
    accuracy = float(correct_count) / len(c_pred)

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} - Accuracy: {accuracy * 100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


# Main code.
def train_har(args):
    if args.train_data == 'ucf101':
        args.num_classes = 101
    elif args.val_data == 'hmdb51':
        args.num_classes = 51
        
    # Print relevant parameters.
    for k, v in args.__dict__.items():
        print(f'{k} : {v}')
    # Empty cuda cache.
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()

    models = []
    criterions = []

    save_dir = os.path.join('ckpt/'+args.model, args.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        

    # Load in correct model file.
    if args.saved_model:
        saved_model_file = args.saved_model
        if args.task == 'har':
            if os.path.exists(saved_model_file):
                ft_model = get_har_model(args)
                saved_dict = torch.load(saved_model_file, weights_only=False)
                try:
                    model_dict = {k.replace('module.', ''): v for k, v in saved_dict.items()}
                    ft_model.load_state_dict(model_dict, strict=True)
                except:
                    model_dict = {k.replace('module.', ''): v for k, v in saved_dict.items() if 'fc' not in k}
                    ft_model.load_state_dict(model_dict, strict=True)
                epoch1 = 1
            else:
                print('*********************************************************************')
                print(f'No such model exists: {saved_model_file} :(')
                print('*********************************************************************')
                epoch1 = 1
        elif args.task == 'pri':
            if os.path.exists(saved_model_file):
                ft_model = get_har_model(args)
                saved_dict = torch.load(saved_model_file, weights_only=False)
                saved_dict = {k.replace('module.', ''): v for k, v in saved_dict['ft_model_state_dict'].items() if 'fc' not in k}
                ft_model.load_state_dict(saved_dict, strict=False)
                epoch1 = 1
            else:
                print('*********************************************************************')
                print(f'No such model exists: {saved_model_file} :(')
                print('*********************************************************************')
                epoch1 = 1
            
    elif args.ckpt_model:
        ckpt_file = args.ckpt_model
        
        if os.path.exists(ckpt_file):
            ft_model = get_har_model(args)
            saved_dict = torch.load(ckpt_file, weights_only=False)
            saved_dict = {k.replace('module.', ''): v for k, v in saved_dict['ft_model_state_dict'].items()}
            ft_model.load_state_dict(saved_dict, strict=True)
            epoch1 = torch.load(ckpt_file, weights_only=False)['epoch']
        else:
            print('*********************************************************************')
            print(f'No such model exists: {ckpt_file} :(')
            print('*********************************************************************')
            epoch1 = 1
        
    else:
        print(f'Training from scratch! ')
        # Load in fresh init ft_model.
        ft_model = get_har_model(args)
        epoch1 = 1

    models.append(ft_model)

    # hide
    if args.hide:
        if args.hide_model == 'lfvsn':
            opt = option.parse('hide/LF_VSN/options/train/train_LF-VSN_1video.yml', is_train=False)
            fh_model = define_G_v2(opt)
            load_network('hide/LF_VSN/param/LF-VSN_1video_hiding_250k.pth', fh_model)
        elif args.hide_model == 'hinet':
            fh_model = HiModel()
            init_model(fh_model)
            state_dicts = torch.load('hide/HiNet/param/model.pt')
            network_state_dict = {k.replace('module.model', 'model'): v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
            fh_model.load_state_dict(network_state_dict)
        elif args.hide_model == 'wengnet':
            fh_model = wengnet()
            fh_model.load_state_dict(torch.load('hide/wengnet/param/weng_checkpoint_2975.pt'))
        elif args.hide_model == 'hidden':
            fh_model = HiDDeN()
            fh_model.load_state_dict(torch.load('hide/hidden/param/hidden_checkpoint_2850.pt'))
            
        fh_model.eval()
        for param in fh_model.parameters():  # 隐写网络不更新参数
            param.requires_grad = False
        dwt = common.DWT()
        iwt = common.IWT()
        models.append(fh_model)
        models.append(dwt)
        models.append(iwt)

    # Init loss function.
    ce = nn.CrossEntropyLoss()
    criterions.append(ce)
    if 'pro' in args.model:
        mse = nn.MSELoss()
        criterions.append(mse)

    scaler = GradScaler()

    device_name = f'cuda:{args.devices[0]}'
    print(f'Device name is {device_name}')
    if len(args.devices) > 1:
        print(f'Multiple GPUS found!')
        for i, model in enumerate(models):
            model = nn.DataParallel(model, device_ids=args.devices)
            model.cuda()
            models[i] = model
        for i, criterion in enumerate(criterions):
            criterion.cuda()
            criterions[i] = criterion
    else:
        print('Only 1 GPU is available')
        for i, model in enumerate(models):
            model.to(device=torch.device(device_name))
            models[i] = model
        for i, criterion in enumerate(criterions):
            criterion.to(device=torch.device(device_name))
            criterions[i] = criterion

    # Select optimizer.
    if args.opt_type == 'adam':
        optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)
    elif args.opt_type == 'adamw':
        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)
    elif args.opt_type == 'sgd':
        optimizer = torch.optim.SGD(ft_model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {args.opt_type} not yet implemented.')

    modes = list(range(args.num_modes))
    cropping_facs = args.cropping_facs

    val_array = [i for i in range(1, args.num_epochs)]

    print(f'Base learning rate {args.learning_rate}')

    best_score = 0
    best_acc = 0
    train_loss = 1000
    learning_rate = args.learning_rate
    orig_learning_rate = learning_rate
    scheduler_epoch = 0
    scheduler_step = 1
    ckpt = 0
    
    if args.ckpt_model:
        saved_dict = torch.load(ckpt_file, weights_only=False)
        optimizer.load_state_dict(saved_dict['optimizer'])
        scaler.load_state_dict(saved_dict['amp_scaler'].state_dict())

    for epoch in range(epoch1, args.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()

        train_dataset = get_train_data(args=args, shuffle=True, data_percentage=args.data_percentage)
        # train_dataset = single_train_dataloader_hmdb(params=params, shuffle=True, data_percentage=params.data_percentage, action_name=action_name)

        if epoch == epoch1:
            print(f'Train dataset length: {len(train_dataset)}')
            print(f'Train dataset steps per epoch: {len(train_dataset) / args.batch_size}')

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_train,
            pin_memory=args.pin_memory)

        # Warmup/LR scheduler.
        if args.lr_scheduler == 'cosine':
            learning_rate = args.cosine_lr_array[epoch - 1] * orig_learning_rate
        elif args.warmup and epoch - 1 < len(args.warmup_array):
            learning_rate = args.warmup_array[epoch - 1] * orig_learning_rate
        elif args.lr_scheduler == 'loss_based':
            if 0.5 <= train_loss < 1.0:
                if args.learning_rate < 1e-3:
                    learning_rate = args.learning_rate
                else:
                    learning_rate = 1e-3
            elif 0.1 <= train_loss < 0.5:
                if epoch > 100 and args.beta > 0.3 and args.hide:
                    learning_rate = 5e-5
                else:
                    learning_rate = 1e-4
            elif train_loss < 0.1:
                learning_rate = 5e-5
        elif args.lr_scheduler == 'patience_based':
            if scheduler_epoch == args.lr_patience:
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                print(
                    f'Dropping learning rate to {learning_rate / (args.lr_reduce_factor ** scheduler_step)} at epoch {epoch}.')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                learning_rate = orig_learning_rate / (args.lr_reduce_factor ** scheduler_step)
                scheduler_epoch = 0
                scheduler_step += 1
        else:
            learning_rate = orig_learning_rate

        if args.ckpt_model and ckpt==0:
            learning_rate = optimizer.param_groups[0]['lr']
            ckpt += 1

        ft_model, train_loss, scaler = train_epoch(epoch, train_dataloader, models, criterions, optimizer,
                                                   learning_rate, scaler, device_name, args)

        if train_loss < best_score:
            best_score = train_loss
            scheduler_epoch = 0
        else:
            scheduler_epoch += 1
            
        # if epoch % 5 == 0:
        #     ft_model.eval()

        # Validation epoch.
        if (epoch % args.val_int == 0) or (train_loss < 0.06 and args.val_data == 'ucf101') or (train_loss < 0.31 and args.val_data == 'hmdb51'):
        # if epoch % args.val_int == 0 or train_loss < 0.1:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    validation_dataset = get_val_data(args=args, shuffle=True, data_percentage=1.0, mode=mode, cropping_factor=cropping_fac)

                    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_workers, drop_last=False,
                                                       collate_fn=collate_val)
                    if val_iter == 0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset) / args.batch_size}')

                    pred_dict, label_dict, accuracy, loss = val_epoch(epoch, mode, cropping_fac, pred_dict, label_dict,
                                                                      validation_dataloader, models, criterions,
                                                                      device_name, args)
                    val_losses.append(loss)

                    predictions = np.zeros((len(list(pred_dict.keys())), args.num_classes))
                    ground_truth = []
                    for entry, key in enumerate(pred_dict.keys()):
                        predictions[entry] = np.mean(pred_dict[key], axis=0)

                    for key in label_dict.keys():
                        ground_truth.append(label_dict[key])

                    pred_array = np.flip(np.argsort(predictions, axis=1),
                                         axis=1)  # Prediction with the most confidence is the first element here.
                    c_pred = pred_array[:, 0]

                    correct_count = np.sum(c_pred == ground_truth)
                    accuracy_all = float(correct_count) / len(c_pred)
                    print(
                        f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]}, is {accuracy_all * 100:.3f}%')
            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())), args.num_classes))
            ground_truth = []

            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1),
                                 axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:, 0]

            correct_count = np.sum(c_pred == ground_truth)
            accuracy = float(correct_count) / len(c_pred)
            print(f'Val loss for epoch {epoch} is {val_loss}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            print(f'Overall Accuracy is for epoch {epoch} is {accuracy * 100:.3f}%')

            if accuracy > best_acc:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {args.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join('ckpt/'+args.model, args.run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(accuracy)[:6]}.pth')
                states = {
                    'epoch': epoch + 1,
                    'amp_scaler': scaler,
                    'ft_model_state_dict': ft_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                # torch.save(states, save_file_path)
                best_acc = accuracy

        # Temp saving.
        save_dir = os.path.join('ckpt/'+args.model, args.run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'amp_scaler': scaler,
            'ft_model_state_dict': ft_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {int(taken)}s')
        print()
        # if args.lr_scheduler != 'cosine' and learning_rate < 1e-12 and epoch > 10:
        #     print(f'Learning rate is very low now, stopping the training.')
        #     break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--params", dest='params', type=str, required=False, default='action_training/params_action.py',
                        help='params')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None,
                        help='devices should be a list')

    args = parser.parse_args()
    # if os.path.exists(args.params):
    #     params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
    #     print(f'{args.params} is loaded as parameter file.')
    # else:
    #     print(f'{args.params} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    # seed_torch(args.seed)
    train_har(args)
