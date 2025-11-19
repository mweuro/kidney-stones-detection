import os
import random
from typing import Literal


def find_jpg_files(root_dir: str) -> list:
    jpgs = []
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg')):
                jpgs.append(os.path.join(dirpath, fname))
    return jpgs


def get_random_image_id(root_dir: str) -> tuple[str, Literal['train', 'valid', 'test']]:
    image_files = find_jpg_files(root_dir)
    random_image_file = random.choice(image_files)
    image_id = os.path.splitext(random_image_file.split(os.sep)[-1])[0]
    dataset = random_image_file.split(os.sep)[-3]  # Assuming structure root/dataset/images/filename.jpg
    return image_id, dataset


def get_original_image_path(image_id: str, dataset: Literal['train', 'valid', 'test']) -> str:
    return os.path.join("data", dataset, "images", f"{image_id}.jpg")


def get_processed_image_path(image_id: str, dataset: Literal['train', 'valid', 'test']) -> str:
    return os.path.join("data_preprocessed", dataset, "images", f"{image_id}.jpg")