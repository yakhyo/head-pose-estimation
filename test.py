import os
import argparse
import torch
import numpy as np
import cv2

from models import resnet
from utils import datasets, helpers
from utils.helpers import get_dataset


def parse_args():
    """Parse input arguments for head pose estimation evaluation"""
    parser = argparse.ArgumentParser(description='Head pose estimation evaluation.')

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data/AFLW2000', help='Directory path for data.')
    parser.add_argument('--dataset', type=str, default='AFLW2000', help='Dataset type.')

    # Data loading params
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')

    # Model weights
    parser.add_argument('--weights', type=str, default='', help='Path to model weight for evaluation.')

    return parser.parse_args()


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
        euler = helpers.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi

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

    print(
        f'Yaw: {yaw_error / total:.4f} ',
        f'Pitch: {pitch_error / total:.4f} ',
        f'Roll: {roll_error / total:.4f} ',
        f'MAE: {(yaw_error + pitch_error + roll_error) / (total * 3):.4f}'
    )


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading test data.')
    test_dataset, test_loader = get_dataset(params, train=False)

    model = resnet.resnet18(num_classes=6)
    if os.path.exists(params.weights):
        model.load_state_dict(torch.load(params.weights, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Model weight not found at {params.weights}")
    model.to(device)
    evaluate(params=params, model=model, data_loader=test_loader, device=devi)


if __name__ == '__main__':
    args = parse_args()
    main(args)
