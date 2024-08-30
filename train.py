
import time
import os
import argparse

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from models import resnet
from utils.loss import GeodesicLoss
from utils import datasets


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
    parser.add_argument('--dataset', type=str, default='Pose_300W_LP', help='Dataset type.')

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
    epoch
) -> None:
    pass


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('output'):
        os.makedirs('output')

    summary_name = '{}_{}'.format('resnet18', int(time.time()), args.batch_size)

    if not os.path.exists('output/{}'.format(summary_name)):
        os.makedirs('output/{}'.format(summary_name))

    model = resnet.resnet18(num_classes=6)

    if not args.output == '':
        saved_state_dict = torch.load(args.output)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pose_dataset = datasets.getDataset(args.dataset, args.data, transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    model.to(device)
    criterion = GeodesicLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    best_train_loss = float('inf')

    print('Starting training.')
    for epoch in range(args.num_epochs):
        loss_sum = 0.0
        iter = 0
        for i, (images, labels, _) in enumerate(train_loader):
            iter += 1
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calc loss
            loss = criterion(labels, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f' %
                      (epoch+1, args.num_epochs, i+1, len(pose_dataset)//args.batch_size, loss.item()))

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = loss_sum / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Average Training Loss: {avg_train_loss:.6f}')

        # Save the last checkpoint
        checkpoint_path = os.path.join('output', summary_name, 'checkpoint.ckpt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # Save the best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join('output', summary_name, 'best_model.pt'))

    print('Training completed.')
