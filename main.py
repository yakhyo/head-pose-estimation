import os
import time
import logging
import argparse

import numpy as np
from PIL import Image

import torch


from models import (
    resnet18,
    resnet34,
    resnet50,
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large
)

from utils.loss import GeodesicLoss
from utils.datasets import get_dataset
from utils.general import random_seed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """head pose estimation training arguments"""
    parser = argparse.ArgumentParser(description='Head pose estimation training.')

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data/300W_LP', help='Directory path for data.')
    parser.add_argument('--dataset', type=str, default='300W', help='Dataset name.')

    # Model and training configuration
    parser.add_argument('--num-epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Network architecture, currently available: resnet18/34/50, mobilenetv2"
    )
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate.')
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue training.")

    # Scheduler configuration
    parser.add_argument(
        '--scheduler',
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

    # Output path
    parser.add_argument('--output', type=str, default='output', help='Path of model output.')

    return parser.parse_args()


def get_model(arch, num_classes=6, pretrained=True):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=num_classes)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=num_classes)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_small":
        model = mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_large":
        model = mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def train_one_epoch(
    params,
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    scheduler=None
):
    model.train()
    loss_sum = 0.0
    for idx, (images, labels, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if (idx + 1) % 100 == 0:
            log_message = (
                f'Epoch [{epoch + 1}/{params.num_epochs}], '
                f'Iter [{idx + 1}/{len(data_loader.dataset) // params.batch_size}] '
                f'Loss: {loss.item():.6f}'
            )
            logging.info(log_message)

    if scheduler is not None:
        scheduler.step()

    avg_train_loss = loss_sum / len(data_loader)
    log_message = f'Epoch [{epoch + 1}/{params.num_epochs}], Average Training Loss: {avg_train_loss:.6f}'
    logging.info(log_message)

    return avg_train_loss


def main(params):
    random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(params.output):
        os.makedirs(params.output)

    summary_name = '{}_{}'.format(params.arch, int(time.time()), params.batch_size)

    if not os.path.exists(f'{params.output}/{summary_name}'):
        os.makedirs(f'{params.output}/{summary_name}')

    model = get_model(params.arch, num_classes=6)
    model.to(device)

    criterion = GeodesicLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.lr)

    start_epoch = 0
    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location=device, weights_only=True)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt['epoch']
        logging.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch + 1}')

    logging.info('Loading training data.')
    train_dataset, train_loader = get_dataset(params)

    if params.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        scheduler = None

    best_train_loss = float('inf')

    logging.info('Starting training.')

    for epoch in range(start_epoch, params.num_epochs):
        avg_train_loss = train_one_epoch(
            params=params,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scheduler=scheduler
        )
        # Save the last checkpoint
        checkpoint_path = os.path.join(params.output, summary_name, 'checkpoint.ckpt')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Save the best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(params.output, summary_name, 'best_model.pt'))

    logging.info('Training completed.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
