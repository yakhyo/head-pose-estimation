import os
import time
import argparse

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from models import get_model

from utils.loss import GeodesicLoss, GeodesicAndFrobeniusLoss
from utils.datasets import Pose300W, AFLW2000
from utils.general import (
    LOGGER,
    setup_seed,
    reduce_tensor,
    save_on_master,
    init_distributed_mode,
    AverageMeter,
    EarlyStopping,
    compute_euler_angles_from_rotation_matrices
)


def parse_args():
    """head pose estimation training arguments"""
    parser = argparse.ArgumentParser(description='Head pose estimation training.')

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data', help='Directory path for data.')

    # Model and training configuration
    parser.add_argument('--epochs', type=int, default=80, help='Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument(
        "--network",
        type=str,
        default="resnet18",
        help="Network architecture, currently available: resnet18/34/50, mobilenetv2"
    )
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate.')
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue training.")

    # lr_scheduler configuration
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='MultiStepLR',
        choices=['StepLR', 'MultiStepLR'],
        help='Learning rate scheduler type.'
    )
    parser.add_argument('--step-size', type=int, default=10, help='Period of learning rate decay for StepLR.')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Multiplicative factor of learning rate decay for StepLR and ExponentialLR.'
    )
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 20],
        help='List of epoch indices to reduce learning rate for MultiStepLR (ignored if StepLR is used).'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Frequency (in batches) for printing training progress. Default: 100.'
    )

    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')

    # Output path
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights',
        help='Path to save model checkpoints. Default: `weights`.'
    )
    return parser.parse_args()


def load_data(train_dir, eval_dir, params):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    LOGGER.info("Loading training data...")
    train_dataset = Pose300W(train_dir, transform=train_transform)

    LOGGER.info("Loading evaluation data...")
    eval_dataset = AFLW2000(eval_dir, transform=eval_transform)

    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(eval_dataset)

    return train_dataset, eval_dataset, train_sampler, test_sampler


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    params
) -> None:
    model.train()
    losses = AverageMeter("Avg Loss", ":6.3f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    last_batch_idx = len(data_loader) - 1

    start_time = time.time()
    for batch_idx, (images, labels, _) in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(labels, outputs)

        if params.distributed:
            # reduce_tensor is used in distributed training to aggregate metrics (e.g., loss, accuracy)
            # across multiple GPUs. It ensures all devices contribute to the final metric computation.
            reduced_loss = reduce_tensor(loss, params.world_size)
        else:
            reduced_loss = loss

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update metrics
        losses.update(reduced_loss.item(), images.size(0))
        batch_time.update(time.time() - start_time)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Reset start time for the next batch
        start_time = time.time()

        if batch_idx % params.print_freq == 0 or last_batch:
            lr = optimizer.param_groups[0]['lr']
            log = (
                f'Epoch: [{epoch+1}/{params.epochs}][{batch_idx:04d}/{len(data_loader):04d}] '
                f'Loss: {losses.avg:6.3f}, '
                f'LR: {lr:.7f} '
                f'Time: {batch_time.avg:4.3f}s'
            )
            LOGGER.info(log)

    # End-of-epoch summary
    log = (
        f'Epoch: [{epoch+1}/{params.epochs}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )
    LOGGER.info(log)


@torch.no_grad()
def evaluate(params, model, data_loader, device):
    model.eval()
    total = 0
    yaw_error = pitch_error = roll_error = 0.0
    v1_err = v2_err = v3_err = 0.0

    for images, r_label, cont_labels, name in data_loader:
        images = images.to(device)
        total += cont_labels.size(0)

        R_gt = r_label

        p_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
        y_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
        r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

        R_pred = model(images)
        euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi

        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()

        R_pred = R_pred.cpu()
        v1_err += torch.sum(torch.acos(torch.clamp(torch.sum(R_gt[:, 0] * R_pred[:, 0], dim=1), -1, 1)) * 180 / np.pi)
        v2_err += torch.sum(torch.acos(torch.clamp(torch.sum(R_gt[:, 1] * R_pred[:, 1], dim=1), -1, 1)) * 180 / np.pi)
        v3_err += torch.sum(torch.acos(torch.clamp(torch.sum(R_gt[:, 2] * R_pred[:, 2], dim=1), -1, 1)) * 180 / np.pi)

        pitch_error += torch.sum(torch.min(torch.stack([
            torch.abs(p_gt_deg - p_pred_deg),
            torch.abs(p_pred_deg + 360 - p_gt_deg),
            torch.abs(p_pred_deg - 360 - p_gt_deg),
            torch.abs(p_pred_deg + 180 - p_gt_deg),
            torch.abs(p_pred_deg - 180 - p_gt_deg)
        ]), dim=0)[0])
        yaw_error += torch.sum(torch.min(torch.stack([
            torch.abs(y_gt_deg - y_pred_deg),
            torch.abs(y_pred_deg + 360 - y_gt_deg),
            torch.abs(y_pred_deg - 360 - y_gt_deg),
            torch.abs(y_pred_deg + 180 - y_gt_deg),
            torch.abs(y_pred_deg - 180 - y_gt_deg)
        ]), dim=0)[0])
        roll_error += torch.sum(torch.min(torch.stack([
            torch.abs(r_gt_deg - r_pred_deg),
            torch.abs(r_pred_deg + 360 - r_gt_deg),
            torch.abs(r_pred_deg - 360 - r_gt_deg),
            torch.abs(r_pred_deg + 180 - r_gt_deg),
            torch.abs(r_pred_deg - 180 - r_gt_deg)
        ]), dim=0)[0])

    LOGGER.info(
        f'Yaw: {yaw_error / total:.4f} '
        f'Pitch: {pitch_error / total:.4f} '
        f'Roll: {roll_error / total:.4f} '
        f'MAE: {(yaw_error + pitch_error + roll_error) / (total * 3):.4f}'
    )
    return (yaw_error + pitch_error + roll_error) / (total * 3)


def main(params):
    setup_seed()
    init_distributed_mode(params)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(params.save_path, params.network)
    os.makedirs(output_path, exist_ok=True)

    model = get_model(params.network, num_classes=6, pretrained=True)
    model.to(device)

    model_without_ddp = model
    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        model_without_ddp = model.module

    criterion = GeodesicLoss()
    # criterion = GeodesicAndFrobeniusLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.lr)

    # Learning rate scheduler
    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {params.lr_scheduler}")

    start_epoch = 0
    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location=device, weights_only=True)

        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt['epoch']
        LOGGER.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch + 1}')

    # Datasets and DataLoaders
    train_dir = os.path.join(params.data, "300W_LP")
    val_dir = os.path.join(params.data, "AFLW2000")

    train_dataset, val_dataset, train_sampler, test_sampler = load_data(train_dir, val_dir, params)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=params.batch_size,
        sampler=test_sampler,
        num_workers=params.num_workers,
        pin_memory=True
    )

    best_angular_error = float("inf")
    mean_angular_error = float("inf")
    early_stopping = EarlyStopping(patience=0)

    LOGGER.info('Starting training.')

    for epoch in range(start_epoch, params.epochs):
        if params.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            params=params,
        )
        lr_scheduler.step()
        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': params
        }
        save_on_master(checkpoint, os.path.join(output_path, 'last_checkpoint.ckpt'))

        if params.local_rank == 0:
            mean_angular_error = evaluate(params, model_without_ddp, val_loader, device)

        if early_stopping(epoch, mean_angular_error):
            break

        # Save the best checkpoint based on training loss
        if mean_angular_error < best_angular_error:
            best_angular_error = mean_angular_error
            save_on_master(checkpoint, os.path.join(output_path, 'best_checkpoint.ckpt'))
            LOGGER.info(
                f"New best mean angular error: {best_angular_error:.4f}."
                f"Model saved to {output_path} with `_best` postfix."
            )

        LOGGER.info(
            f"Epoch {epoch + 1} completed. Latest model saved to {output_path} with `_last` postfix."
            f"Best mean angular error: {best_angular_error:.4f}"
        )

    LOGGER.info('Training completed.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
