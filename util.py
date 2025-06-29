import sys
import yaml
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np
from collections import OrderedDict
import os
import random
import torch
from torchvision import transforms, datasets
from PIL import Image
import math

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.__stdout__
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def read_yaml(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except FileNotFoundError:
        print(f"File {path} not found.")
    except yaml.YAMLError as exc:
        print(f"Error in YAML file {path}: {exc}")
    except Exception as exc:
        print(f"An error occurred while reading the YAML file {path}: {exc}")


def print_yaml_params(data):
    for key, value in data.items():
        print(str(key) + ' : ' + str(value))

def translate_duplicate(img, cropSize=224):
    if min(img.size) < cropSize:
        width, height = img.size

        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)

        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img

def random_rotate(img):
    angle = random.choice([0, 90, 180, 270])
    return img.rotate(angle)


def get_dataset(root, classes):
    dataset_list = []
    transform = transforms.Compose([
            transforms.Lambda(translate_duplicate),
            transforms.RandomCrop((224, 224), None),
            #transforms.Lambda(random_rotate),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    for category in classes:
        category_path = os.path.join(root, category)
        sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
        dataset_list.append(sub_dataset)
    return torch.utils.data.ConcatDataset(dataset_list)


def get_dataset_test(root, classes):
    dataset_list = []
    transform = transforms.Compose([
            transforms.Lambda(translate_duplicate),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    for category in classes:
        category_path = os.path.join(root, category)
        sub_dataset = datasets.ImageFolder(root=category_path, transform=transform)
        dataset_list.append(sub_dataset)
    return torch.utils.data.ConcatDataset(dataset_list)

def get_bal_sampler(dataset):
    targets = []
    for sub_dataset in dataset.datasets:
        targets.extend(sub_dataset.targets)
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    weights = torch.from_numpy(weights.astype(np.float32))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
