import os
import cv2
import random
import numpy as np
from scipy import io
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.general import get_rotation_matrix


def load_filenames(root_dir):
    filenames = []
    removed_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                mat_path = os.path.join(root, file.replace('.jpg', '.mat'))
                label = io.loadmat(mat_path)
                pitch, yaw, roll = label['Pose_Para'][0][:3]

                # Convert radians to degrees
                pitch *= 180 / np.pi
                yaw *= 180 / np.pi
                roll *= 180 / np.pi

                # Only add the file if the conditions are met
                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    filenames.append(os.path.join(root, file[:-4]))
                else:
                    removed_count += 1

    return filenames, removed_count


class Pose300W(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames, removed_items = load_filenames(root)
        print(f"Pose300W: {removed_items} items removed from dataset that have an angle > 99 degrees. Loaded {len(self)} files.")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = f"{filename}.jpg"
        mat_path = f"{filename}.mat"

        image = Image.open(img_path).convert("RGB")
        lbl = io.loadmat(mat_path)
        pt2d = lbl['pt2d']
        pitch, yaw, roll = lbl['Pose_Para'][0][:3]

        x_min, x_max = np.min(pt2d[0, :]), np.max(pt2d[0, :])
        y_min, y_max = np.min(pt2d[1, :]), np.max(pt2d[1, :])

        # k calculation and crop adjustments
        k = random.uniform(0.2, 0.4)
        dx = 0.6 * k * (x_max - x_min)
        dy = 0.6 * k * (y_max - y_min)
        x_min, x_max = x_min - dx, x_max + dx
        y_min, y_max = y_min - 2 * dy, y_max + dy

        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
        image = image.crop((x_min, y_min, x_max, y_max))

        if random.random() < 0.5:
            yaw, roll = -yaw, -roll
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.05:
            image = image.filter(ImageFilter.BLUR)

        rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, rotation_matrix, filename

    def __len__(self):
        return len(self.filenames)


class AFLW2000(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames, removed_items = load_filenames(root)
        print(f"AFLW200: {removed_items} items removed from dataset that have an angle > 99 degrees. Loaded {len(self)} files.")

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = f"{filename}.jpg"
        mat_path = f"{filename}.mat"

        image = Image.open(img_path).convert("RGB")
        lbl = io.loadmat(mat_path)
        pt2d = lbl['pt2d']
        pitch, yaw, roll = lbl['Pose_Para'][0][:3]

        x_min, x_max = min(pt2d[0, :]), max(pt2d[0, :])
        y_min, y_max = min(pt2d[1, :]), max(pt2d[1, :])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
        image = image.crop((x_min, y_min, x_max, y_max))

        rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

        labels = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, rotation_matrix, labels, filename

    def __len__(self):
        return len(self.filenames)


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext)).convert('RGB')
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length


class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length


class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        yaw = self.y_train[index][0] / 180*np.pi
        pitch = self.y_train[index][1] / 180*np.pi
        roll = self.y_train[index][2] / 180*np.pi

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = helpers.get_rotation_matrix(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length


def get_dataset(params, train=True):

    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if params.dataset == '300W':
        pose_dataset = Pose300W(params.data, transform)
    elif params.dataset == 'AFLW2000':
        pose_dataset = AFLW2000(params.data, transform)
    elif params.dataset == 'BIWI':
        pose_dataset = BIWI(params.data,  transform, train_mode=train)
    elif params.dataset == 'AFLW':
        pose_dataset = AFLW(params.data,  transform)
    elif params.dataset == 'AFW':
        pose_dataset = AFW(params.data,  transform)
    else:
        raise NameError('Error: not a valid dataset name')

    if params.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(pose_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(pose_dataset)
        
    
    data_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=params.batch_size,
        sampler=sampler,
        num_workers=params.num_workers,
        pin_memory=True
    )
    return pose_dataset, data_loader, sampler
