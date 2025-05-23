import argparse
import os
import pdb

import torch
from torch.utils.data import DataLoader
# from pytorch_warmup import LinearWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
from timm.utils import AverageMeter
import time, datetime, random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from logger import create_logger


from yfcc_dataset.yfcc_face_dataset_multilabel_v4 import yfcc_face
from yfcc_face_model import init_model_classification
from L2R_loss import L2R_Loss, Fidelity_Loss

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"


def main_RN50(args, training_config):
    train_dataset = yfcc_face(train_mode=True, cls_face_mode=True, face_scale=1.3)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    val_dataset = yfcc_face(train_mode=False, face_scale=1.3)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    model, _ = init_model_classification(device=DEVICE,
                                      state_dict_path=args.resume_pretrained,
                                      visual="RN50", ranking=True, exif_only=False)

    criterion = Fidelity_Loss()
    criterion_l2r = L2R_Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"], betas=(0.9, 0.98), eps=1e-6,
                                      weight_decay=0.0001)
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=10,
                                     eta_min=1e-7)

    start_time = time.time()
    log_writer = SummaryWriter(log_dir=output_path)
    for epoch in range(training_config["num_epochs"]):

        train_loss, loss_exif, loss_face, train_face = train_one_epoch(train_dataloader, optimizer, criterion,
                                                                       criterion_l2r, model, epoch, training_config,
                                                                       log_writer)
        val_loss = val_one_epoch(val_dataloader, criterion_l2r, model, epoch, training_config, log_writer)
        with open(os.path.join(output_path, 'logtxt.txt'), mode='a') as f:
            f.write(f'Epoch {epoch} - avg_loss: train {train_loss:.4f} exif {loss_exif:.4f} face {loss_face:.4f} '
                    f'val {val_loss:.4f}')
            f.write('\n')
            f.write(f'Epoch {epoch} - avg_acc: face {train_face:.4f}')
            f.write('\n')


        lr_scheduler.step()

        if args.multi_gpu:
            save_model = model.module
        else:
            save_model = model
        checkpoint = {
            'epoch': epoch,
            'model': save_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_scheduler.state_dict()
        }
        save_path = os.path.join(output_path, f'ckpt_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def predict_batch(tensor_batch):
    _, predicted_indices = torch.max(tensor_batch, dim=1)
    return predicted_indices

def predict_batch_onelogit(tensor_batch):
    predicted_indices = torch.zeros_like(tensor_batch)
    predicted_indices[tensor_batch>=0.5] = 1
    return predicted_indices

def accuracy_batch(predicted_indices, true_labels):
    correct = (predicted_indices == true_labels)
    acc = correct.sum().item() / true_labels.size(0)
    return acc


def train_one_epoch(data_loader, optimizer, criterion, criterion_l2r,
                    model, epoch, training_config, log_writer):
    model.train()

    loss_meter = AverageMeter()
    loss_exif_meter = AverageMeter()
    loss_face_meter = AverageMeter()
    acc_meter_face = AverageMeter()
    batch_time = AverageMeter()
    num_steps = len(data_loader)

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        img, labels, _  = batch
        img = img.to(DEVICE)
        iso = labels['iso'].to(DEVICE)
        av = labels['av'].to(DEVICE)
        et = labels['et'].to(DEVICE)
        fl = labels['fl'].to(DEVICE)

        optimizer.zero_grad()

        logits_per_image = model(img)
        logits_iso, logits_av, logits_et, logits_fl, logits_face = logits_per_image

        loss_1 = criterion_l2r(logits_iso, iso)
        loss_2 = criterion_l2r(logits_av, av)
        loss_3 = criterion_l2r(logits_et, et)
        loss_4 = criterion_l2r(logits_fl, fl)
        loss_exif = loss_1 + loss_2 + loss_3 + loss_4

        loss_face = criterion(logits_face.squeeze(1).sigmoid(), labels['face'].to(DEVICE))
        total_loss = loss_exif + loss_face

        total_loss.backward()
        optimizer.step()

        # training acc
        predicted_indices_l5 = predict_batch_onelogit(logits_face.squeeze(1).sigmoid())  # face
        acc_face = accuracy_batch(predicted_indices_l5, labels['face'].to(DEVICE))

        loss_meter.update(total_loss.item(), img.size(0))
        acc_meter_face.update(acc_face, img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{training_config["num_epochs"]}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc_face {acc_meter_face.val:.4f} ({acc_meter_face.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            log_writer.add_scalar('Train/total_loss', loss_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))  # *1000

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, loss_exif_meter.avg, loss_face_meter.avg, acc_meter_face.avg


@torch.no_grad()
def val_one_epoch(data_loader, criterion_l2r,
                  model, epoch, training_config, log_writer, exif_only=False):
    model.eval()

    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    num_steps = len(data_loader)

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        img, labels, _ = batch
        img = img.to(DEVICE)
        iso = labels['iso'].to(DEVICE)
        av = labels['av'].to(DEVICE)
        et = labels['et'].to(DEVICE)
        fl = labels['fl'].to(DEVICE)

        logits_per_image = model(img)
        if not exif_only:
            logits_iso, logits_av, logits_et, logits_fl, logits_face = logits_per_image
        else:
            logits_iso, logits_av, logits_et, logits_fl = logits_per_image
        loss_1 = criterion_l2r(logits_iso, iso)
        loss_2 = criterion_l2r(logits_av, av)
        loss_3 = criterion_l2r(logits_et, et)
        loss_4 = criterion_l2r(logits_fl, fl)
        total_loss = loss_1 + loss_2 + loss_3 + loss_4

        loss_meter.update(total_loss.item(), img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 100 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{training_config["num_epochs"]}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            log_writer.add_scalar('Val/total_loss', loss_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))  # *1000

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} validation takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg




def setup_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument('--multi_gpu', default=False, action='store_true', help='Bool type')

    parser.add_argument('--resume_pretrained', default="./pretrained/wrapper_75_new.pth", type=str)
    parser.add_argument('--output', default='./output/train', type=str)
    parser.add_argument('--name', default='5heads-lr1e5-L2R', type=str)

    # parser.add_argument('--exif_only', action='store_true')

    args = parser.parse_args()

    setup_seed(42)

    output_path = os.path.join(args.output, args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = create_logger(output_dir=output_path, name=f"{args.name}")

    training_config = {
        "lr": args.lr,
        "num_epochs": args.num_epochs,
    }

    main_RN50(args, training_config)
