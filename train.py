import os
import csv
import time
import argparse

import numpy as np
from PIL import Image

import torch

from models import resnet
from utils.loss import GeodesicLoss
from utils import datasets
from utils.helpers import get_dataset


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the 6DRepNet.')

    # Model and training configuration
    parser.add_argument('--num-epochs', type=int, default=80, help='Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=80, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate.')
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")

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

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data/300W_LP', help='Directory path for data.')
    parser.add_argument('--dataset', type=str, default='300W', help='Dataset name.')

    # Output path
    parser.add_argument('--output', type=str, default='', help='Path of model output.')

    return parser.parse_args()


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
    loss_sum = 0.0
    iter = 0
    for idx, (images, labels, _) in enumerate(data_loader):
        iter += 1
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
            print(log_message)

    if scheduler is not None:
        scheduler.step()

    avg_train_loss = loss_sum / len(data_loader)
    log_message = f'Epoch [{epoch + 1}/{params.num_epochs}], Average Training Loss: {avg_train_loss:.6f}'
    print(log_message)

    return avg_train_loss


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('output'):
        os.makedirs('output')

    summary_name = '{}_{}'.format('resnet18', int(time.time()), params.batch_size)

    if not os.path.exists(f'output/{summary_name}'):
        os.makedirs(f'output/{summary_name}')

    model = resnet.resnet18(num_classes=6)
    model.to(device)

    criterion = GeodesicLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.lr)

    if not params.output == '':
        saved_state_dict = torch.load(params.output)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')
    train_dataset, train_loader = get_dataset(params)

    if params.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        scheduler = None

    best_train_loss = float('inf')

    print('Starting training.')

    for epoch in range(params.num_epochs):
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
        checkpoint_path = os.path.join('output', summary_name, 'checkpoint.ckpt')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Save the best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join('output', summary_name, 'best_model.pt'))

    print('Training completed.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
