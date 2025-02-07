import os
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row["image_path"]
        age_class = row["age_class"]
        image_age_list.append([image_path, age_class])
    return image_age_list


class MUFACDataset(Dataset):
    def __init__(
        self, meta_data, image_directory, transform=None, forget=False, retain=False
    ):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }

        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

        # self.data = []
        # self.labels = []
        # for image_path, age_class in self.image_age_list:
        #     img = Image.open(os.path.join(self.image_directory, image_path))
        #     label = self.age_class_to_label[age_class]
        #     if self.transform:
        #         img = self.transform(img)
        #     self.data.append(img)
        #     self.labels.append(label)

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose([transforms.Resize(128), transforms.ToTensor()])


def get_dataset(train=True, forget=False, retain=False, train_transforms=True):
    tfs = None
    if train:
        if train_transforms:
            tfs = get_transforms()
        else:
            tfs = get_transforms(train=False)
        train_meta_data_path = (
            "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
        )
        train_meta_data = pd.read_csv(train_meta_data_path)
        train_image_directory = (
            "./custom_korean_family_dataset_resolution_128/train_images"
        )
        return MUFACDataset(
            train_meta_data, train_image_directory, tfs, forget=forget, retain=retain
        )
    else:
        tfs = get_transforms(train=False)
        test_meta_data_path = (
            "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
        )
        test_meta_data = pd.read_csv(test_meta_data_path)
        test_image_directory = (
            "./custom_korean_family_dataset_resolution_128/val_images"
        )
        return MUFACDataset(test_meta_data, test_image_directory, tfs)


def get_dataloader(train=True, forget=False, retain=False, train_transforms=True):
    dataset = get_dataset(train, forget, retain, train_transforms)
    return DataLoader(dataset, batch_size=64, shuffle=True)
