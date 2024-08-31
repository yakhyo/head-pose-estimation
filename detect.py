import os
import time
import logging
import argparse
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis
from models import resnet18, resnet34, resnet50, mobilenet_v2, SCRFD

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--cam', type=str, default=0, help='Camera device id to use [0]')
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument("--draw-type", type=str, default='cube', help="Draw cube or axis for head pose")
    parser.add_argument('--weights', type=str, required=True, help='Path to head pose estimation model weights')
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")

    return parser.parse_args()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor."""
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(num_classes=6, pretrained=False)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    logging.info('Loading data.')

    assert os.path.isfile("./weights/det_10g.onnx"), f"Face detector model path is not correct"
    face_detector = SCRFD(model_path="./weights/det_10g.onnx")
    logging.info("Face Detection model weights loaded.")

    # Initialize video capture
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Initialize VideoWriter if saving video
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                width = x_max - x_min
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                start = time.time()
                rotation_matrix = model(image)
                logging.info('Head pose estimation: %.2f ms' % ((time.time() - start) * 1000))

                euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                if args.draw_type == "cube":
                    draw_cube(
                        frame,
                        y_pred_deg,
                        p_pred_deg,
                        r_pred_deg,
                        bbox=[x_min, y_min, x_max, y_max],
                        size=width
                    )
                else:
                    draw_axis(
                        frame,
                        y_pred_deg,
                        p_pred_deg,
                        r_pred_deg,
                        bbox=[x_min, y_min, x_max, y_max],
                        size_ratio=0.5
                    )

            if args.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Write the frame to the video file if saving
            if args.output:
                out.write(frame)

    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
