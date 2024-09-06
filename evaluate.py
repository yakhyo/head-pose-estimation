import os
import cv2
import logging
import argparse
import numpy as np

import torch

from models import resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from utils.datasets import get_dataset
from utils.general import compute_euler_angles_from_rotation_matrices

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """Parse input arguments for head pose estimation evaluation"""
    parser = argparse.ArgumentParser(description='Head pose estimation evaluation.')

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data/AFLW2000', help='Directory path for data.')
    parser.add_argument('--dataset', type=str, default='AFLW2000', help='Dataset type.')
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Network architecture, currently available: resnet18/34/50, mobilenetv2"
    )

    # Data loading params
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')

    # Model weights
    parser.add_argument('--weights', type=str, default='', help='Path to model weight for evaluation.')

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


@torch.no_grad()
def evaluate(
    params,
    model,
    data_loader,
    device
):
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

    logging.info(
        f'Yaw: {yaw_error / total:.4f} '
        f'Pitch: {pitch_error / total:.4f} '
        f'Roll: {roll_error / total:.4f} '
        f'MAE: {(yaw_error + pitch_error + roll_error) / (total * 3):.4f}'
    )


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info('Loading test data.')
    test_dataset, test_loader = get_dataset(params, train=False)

    model = get_model(params.arch, num_classes=6, pretrained=False)
    if os.path.exists(params.weights):
        model.load_state_dict(torch.load(params.weights, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Model weight not found at {params.weights}")
    model.to(device)
    evaluate(params=params, model=model, data_loader=test_loader, device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
