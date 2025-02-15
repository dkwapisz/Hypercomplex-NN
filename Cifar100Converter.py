import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CIFAR100_PATH = "cifar-100-python"

OUTPUT_DIR = "datasets"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

def load_cifar100():
    with open(os.path.join(CIFAR100_PATH, "train"), "rb") as f:
        train_data = pickle.load(f, encoding="bytes")
    with open(os.path.join(CIFAR100_PATH, "test"), "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    images = np.concatenate([train_data[b'data'], test_data[b'data']], axis=0)
    labels = np.concatenate([train_data[b'fine_labels'], test_data[b'fine_labels']], axis=0)

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    with open(os.path.join(CIFAR100_PATH, "meta"), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b'fine_label_names']]

    return images, labels, class_names


def save_images(images, labels, class_names, split_ratios):
    train_ratio, val_ratio, test_ratio = split_ratios
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subset), exist_ok=True)

    for class_id, class_name in enumerate(class_names):
        class_indices = np.where(labels == class_id)[0]
        train_idx, temp_idx = train_test_split(class_indices, test_size=(1 - train_ratio), random_state=SEED)
        val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)),
                                             random_state=SEED)

        for subset, indices in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            class_folder = os.path.join(OUTPUT_DIR, subset, class_name)
            os.makedirs(class_folder, exist_ok=True)

            for i, img_idx in enumerate(indices):
                img = Image.fromarray(images[img_idx])
                img.save(os.path.join(class_folder, f"{i}.png"))

images, labels, class_names = load_cifar100()
save_images(images, labels, class_names, (TRAIN_RATIO, VAL_RATIO, TEST_RATIO))
