import os
import cv2
from PIL import Image

from face_detection import RetinaFace
from torchvision import transforms
import torch

import numpy as np
import time
import argparse

from utils.general import compute_euler_angles_from_rotation_matrices, plot_pose_cube, draw_axis
from models import resnet


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--cam', type=str, default=0, help='Camera device id to use [0]')
    parser.add_argument('--snapshot', type=str, default='_epoch_12.tar', help='Name of model snapshot.')
    parser.add_argument('--save_viz', action='store_true', help='Save images with pose cube.')
    parser.add_argument('--output_video', type=str, default='output_video.avi', help='Name of output video file.')

    return parser.parse_args()


transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet.resnet18(num_classes=6)
    state_dict = torch.load(args.snapshot, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    print('Loading data.')

    detector = RetinaFace(gpu_id=0)

    # Initialize video capture
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Initialize VideoWriter if saving video
    if args.save_viz:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("out_video_v2.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector(frame)
            for box, landmarks, score in faces:
                if score < 0.95:
                    continue

                x_min, y_min, x_max, y_max = map(int, box)
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * bbox_height))
                y_min = max(0, y_min - int(0.2 * bbox_width))
                x_max += int(0.2 * bbox_height)
                y_max += int(0.2 * bbox_width)

                image = frame[y_min:y_max, x_min:x_max]
                image = Image.fromarray(image).convert('RGB')
                image = transformations(image).unsqueeze(0).to(device)

                start = time.time()
                R_pred = model(image)
                end = time.time()
                print('Head pose estimation: %.2f ms' % ((end - start) * 1000))

                euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # plot_pose_cube(
                #     frame, y_pred_deg, p_pred_deg, r_pred_deg,
                #     x_min + int(0.5 * (x_max - x_min)),
                #     y_min + int(0.5 * (y_max - y_min)),
                #     size=bbox_width
                # )

                draw_axis(
                    frame,
                    y_pred_deg,
                    p_pred_deg,
                    r_pred_deg,
                    bbox=[x_min, y_min, x_max, y_max],
                    size_ratio=0.5
                )

            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) == 27:  # Exit on ESC key
            #     break

            # Write the frame to the video file if saving
            if args.save_viz:
                out.write(frame)

    cap.release()
    if args.save_viz:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
