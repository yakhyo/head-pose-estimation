import os
import cv2
import random
import logging
import numpy as np
import scipy.io as sio

import torch
import torch.distributed as distributed

from typing import List


# Custom filter to restrict logs to the main process
class MainProcessFilter(logging.Filter):
    def filter(self, record):
        return is_main_process()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger()
LOGGER.addFilter(MainProcessFilter())


class EarlyStopping:
    """
    Early stopping utility to stop training when the monitored metric stops improving.
    Optimized for minimizing error (e.g., validation loss).
    """

    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience if patience != 0 else float("inf")
        self.min_delta = min_delta
        self.best_fitness = float("inf")  # Initialize to a large value for minimization
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, fitness):
        """
        Checks if training should stop.
        Args:
            epoch (int): Current epoch number.
            fitness (float): Current metric value to monitor (e.g., error or loss).
        """
        if fitness < self.best_fitness - self.min_delta:  # Improvement if fitness decreases
            self.best_fitness = fitness
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, "
                f"lowest metric: {self.best_fitness:.4f}"
            )
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def is_dist_avail_and_initialized():
    if not distributed.is_available():
        return False
    if not distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """
    Initializes the distributed mode for multi-GPU training.

    Args:
        args: Argument parser object with the necessary attributes.
    """
    # Check for distributed environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Distributed mode not enabled. Falling back to single process.")
        args.distributed = False
        return

    args.distributed = True

    # Set the device
    torch.cuda.set_device(args.local_rank)
    print(f"| Distributed initialization (rank {args.rank}): env://", flush=True)

    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank
    )
    setup_for_distributed(args.rank == 0)


def reduce_tensor(tensor, n):
    """Getting the average of tensors over multiple GPU devices
    Args:
        tensor: input tensor
        n: world size (number of gpus)
    Returns:
        reduced tensor
    """
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= n
    return rt


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize_vector(v):
    """
    Normalizes a batch of vectors to have unit length.

    Args:
        v (torch.Tensor): A tensor of shape (batch_size, n) representing a batch of n-dimensional vectors.

    Returns:
        torch.Tensor: A tensor of the same shape as input, where each vector is normalized to unit length.
    """
    v_mag = torch.norm(v, p=2, dim=1, keepdim=True)  # Compute the magnitude of each vector, p=2 - l2 norm Euclidean
    v_mag = torch.clamp(v_mag, min=1e-8)  # Avoid division by zero
    v_normalized = v / v_mag  # Normalize each vector

    return v_normalized


def cross_product(u, v):
    """
    Computes the cross product of two batches of 3D vectors.

    Args:
        u (torch.Tensor): A tensor of shape (batch_size, 3) representing the first set of 3D vectors.
        v (torch.Tensor): A tensor of shape (batch_size, 3) representing the second set of 3D vectors.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing the cross product of each pair of vectors.
    """
    i_component = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j_component = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k_component = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    cross_prod = torch.cat((i_component.unsqueeze(1), j_component.unsqueeze(1), k_component.unsqueeze(1)), dim=1)

    return cross_prod


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Computes a batch of 3x3 rotation matrices from 6D orthogonal representation.

    Args:
        poses (torch.Tensor): A tensor of shape (batch_size, 6) containing the 6D vectors.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3, 3) containing the rotation matrices.
    """
    x_raw = poses[:, 0:3]  # First 3D vector
    y_raw = poses[:, 3:6]  # Second 3D vector

    x = normalize_vector(x_raw)  # Normalize the first vector
    z = cross_product(x, y_raw)  # Compute the cross product of x and y_raw to get z
    z = normalize_vector(z)  # Normalize z
    y = cross_product(z, x)  # Compute y by crossing z and x

    # Reshape x, y, z to (batch_size, 3, 1) and concatenate them to form rotation matrices
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)

    rotation_matrix = torch.cat((x, y, z), dim=2)  # Concatenate along the last dimension

    return rotation_matrix


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    """
    Computes the Euler angles (x, y, z) from a batch of 3x3 rotation matrices.

    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (batch_size, 3, 3) containing  the rotation matrices.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing the Euler angles (x, y, z) for each rotation matrix in the batch.
    """
    batch_size = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)

    is_singular = sy < 1e-6

    x_angle = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y_angle = torch.atan2(-R[:, 2, 0], sy)
    z_angle = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    x_angle_singular = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    y_angle_singular = torch.atan2(-R[:, 2, 0], sy)
    z_angle_singular = torch.zeros_like(z_angle)

    euler_angles = torch.zeros(batch_size, 3)

    euler_angles[:, 0] = x_angle * (~is_singular) + x_angle_singular * is_singular
    euler_angles[:, 1] = y_angle * (~is_singular) + y_angle_singular * is_singular
    euler_angles[:, 2] = z_angle * (~is_singular) + z_angle_singular * is_singular

    return euler_angles


def get_rotation_matrix(x, y, z):
    """
    Computes the 3D rotation matrix from three rotation angles (in radians).
    The rotation is performed in a right-handed coordinate system, with rotations
    applied in the order: x-axis, y-axis, and z-axis.

    Args:
        x (float): Rotation angle around the x-axis (in radians).
        y (float): Rotation angle around the y-axis (in radians).
        z (float): Rotation angle around the z-axis (in radians).

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing the combined rotation.
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    R = Rz.dot(Ry.dot(Rx))
    return R


def draw_cube(image: np.ndarray, yaw: float, pitch: float, roll: float, bbox: List[int], size: int = 150) -> None:
    """
    Plots a 3D pose cube on a given image based on yaw, pitch, and roll angles.

    All 8 vertices are computed symmetrically around the bounding box center
    so the cube stays centered on the face at any rotation.

    Args:
        image (np.array): The input image where the cube will be drawn.
        yaw (float): The yaw angle in degrees.
        pitch (float): The pitch angle in degrees.
        roll (float): The roll angle in degrees.
        bbox (List[int]): Bounding box coordinates as [x_min, y_min, x_max, y_max].
        size (float, optional): Size of the cube. Defaults to 150.
    """
    yaw_r, pitch_r, roll_r = np.radians([-yaw, pitch, roll])

    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    h = size * 0.5

    cos_y, sin_y = np.cos(yaw_r), np.sin(yaw_r)
    cos_p, sin_p = np.cos(pitch_r), np.sin(pitch_r)
    cos_r, sin_r = np.cos(roll_r), np.sin(roll_r)

    ex = np.array([cos_y * cos_r, cos_p * sin_r + cos_r * sin_p * sin_y])
    ey = np.array([-cos_y * sin_r, cos_p * cos_r - sin_p * sin_y * sin_r])
    ez = np.array([sin_y, -cos_y * sin_p])

    center = np.array([cx, cy])

    def _pt(v: np.ndarray) -> tuple:
        return (int(v[0]), int(v[1]))

    f0 = center + h * (-ex - ey - ez)
    f1 = center + h * (+ex - ey - ez)
    f2 = center + h * (+ex + ey - ez)
    f3 = center + h * (-ex + ey - ez)
    b0 = center + h * (-ex - ey + ez)
    b1 = center + h * (+ex - ey + ez)
    b2 = center + h * (+ex + ey + ez)
    b3 = center + h * (-ex + ey + ez)

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)

    # Front face at head (red)
    cv2.line(image, _pt(f0), _pt(f1), red, 2)
    cv2.line(image, _pt(f1), _pt(f2), red, 2)
    cv2.line(image, _pt(f2), _pt(f3), red, 2)
    cv2.line(image, _pt(f3), _pt(f0), red, 2)

    # Back face in looking direction (green)
    cv2.line(image, _pt(b0), _pt(b1), green, 2)
    cv2.line(image, _pt(b1), _pt(b2), green, 2)
    cv2.line(image, _pt(b2), _pt(b3), green, 2)
    cv2.line(image, _pt(b3), _pt(b0), green, 2)

    # Side edges (blue)
    cv2.line(image, _pt(f0), _pt(b0), blue, 2)
    cv2.line(image, _pt(f1), _pt(b1), blue, 2)
    cv2.line(image, _pt(f2), _pt(b2), blue, 2)
    cv2.line(image, _pt(f3), _pt(b3), blue, 2)


def draw_axis(image: np.ndarray, yaw: float, pitch: float, roll: float, bbox: List[int], size_ratio: float = 0.5) -> None:
    """
    Draws 3D coordinate axes on a 2D image based on yaw, pitch, and roll angles.

    Args:
        image (numpy.ndarray): The image to draw on.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
        roll (float): Roll angle in degrees.
        bbox (List[int]): Bounding box [x_min, y_min, x_max, y_max].
        size_ratio (float, optional): Scaling factor for the axis length. Defaults to 0.5.
    """
    # Convert angles from degrees to radians
    yaw, pitch, roll = np.radians([-yaw, pitch, roll])

    # Bounding box calculations
    x_min, y_min, x_max, y_max = bbox
    tdx = int(x_min + (x_max - x_min) * 0.5)
    tdy = int(y_min + (y_max - y_min) * 0.5)

    bbox_size = min(x_max - x_min, y_max - y_min)
    size = bbox_size * size_ratio

    # Pre-compute trigonometric values
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    # X-Axis | drawn in red
    x1 = int(size * (cos_yaw * cos_roll) + tdx)
    y1 = int(size * (cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw) + tdy)

    # Y-Axis | drawn in green
    x2 = int(size * (-cos_yaw * sin_roll) + tdx)
    y2 = int(size * (cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll) + tdy)

    # Z-Axis | drawn in blue
    x3 = int(size * sin_yaw + tdx)
    y3 = int(size * (-cos_yaw * sin_pitch) + tdy)

    cv2.line(image, (tdx, tdy), (x1, y1), (0, 0, 255), 2)  # Red (X-axis)
    cv2.line(image, (tdx, tdy), (x2, y2), (0, 255, 0), 2)  # Green (Y-axis)
    cv2.line(image, (tdx, tdy), (x3, y3), (255, 0, 0), 2)  # Blue (Z-axis)
