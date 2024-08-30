import os
import argparse
import torch
import numpy as np
import cv2

from models import resnet
from utils import datasets, helpers
from utils.helpers import get_dataset


def parse_args():
    """Parse input arguments for head pose estimation using 6DRepNet."""
    parser = argparse.ArgumentParser(description='Head pose estimation using 6DRepNet.')

    # Dataset and data paths
    parser.add_argument('--data', type=str, default='data/AFLW2000', help='Directory path for data.')
    parser.add_argument('--dataset', type=str, default='AFLW2000', help='Dataset type.')
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")

    # Model configuration
    parser.add_argument('--weights', type=str, default='', help='Path to model weight for evaluation.')

    # Training configuration
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')

    # Visualization options
    parser.add_argument('--show-viz', action='store_true', help='Show images with pose cube visualization.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading data.')

    test_dataset, test_loader = get_dataset(args, train=False)

    model = resnet.resnet18(num_classes=6)
    model.load_state_dict(torch.load(args.weights))
    model.to(device)
    model.eval()

    total = 0
    yaw_error = pitch_error = roll_error = 0.0
    v1_err = v2_err = v3_err = 0.0

    with torch.no_grad():
        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
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
            v1_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 0] * R_pred[:, 0], dim=1), -1, 1)) * 180 / np.pi
            )
            v2_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 1] * R_pred[:, 1], dim=1), -1, 1)) * 180 / np.pi
            )
            v3_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 2] * R_pred[:, 2], dim=1), -1, 1)) * 180 / np.pi
            )

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

            if args.show_viz:
                name = name[0]
                if args.dataset == 'AFLW2000':
                    cv2_img = cv2.imread(os.path.join(args.data, name + '.jpg'))
                elif args.dataset == 'BIWI':
                    vis = np.uint8(name)
                    cv2_img = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                helpers.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
                cv2.imshow("Test", cv2_img)
                cv2.waitKey(5)
                cv2.imwrite(os.path.join('output/img/', name + '.png'), cv2_img)

    print(
        f'Yaw: {yaw_error / total:.4f} ',
        f'Pitch: {pitch_error / total:.4f} ',
        f'Roll: {roll_error / total:.4f} ',
        f'MAE: {(yaw_error + pitch_error + roll_error) / (total * 3):.4f}'
    )
