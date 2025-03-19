# Import necessary libraries
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import textwrap
import shutil

# Define the path to the dataset
dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'  # Replace with the path to your dataset
dataset_dir = 'dataset/tiny-imagenet'

# Send a GET request to the URL
response = requests.get(dataset_url)
# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded bytes and extract them
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(dataset_dir)
    print('Download and extraction complete!')

# Organize the validation set into subdirectories
val_dir = os.path.join(dataset_dir, 'tiny-imagenet-200', 'val')
val_images_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Check if the val_annotations.txt file exists
if not os.path.exists(val_annotations_file):
    print(f"File not found: {val_annotations_file}")
    exit(1)

with open(val_annotations_file) as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.move(os.path.join(val_images_dir, fn), os.path.join(cls_dir, fn))

shutil.rmtree(val_images_dir)
print('Validation set organized!')