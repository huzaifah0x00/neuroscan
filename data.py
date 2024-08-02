import os
import re

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGES = "dataset/images"
LABELS = "dataset/labels"


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")], key=natural_sort_key)
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")], key=natural_sort_key)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            labels = [line.strip().split() for line in f]

        if self.transform is not None:
            image = self.transform(image)

        if not labels:
            target = {"box": torch.as_tensor([0, 0, 0, 0]), "class_label": torch.tensor(0)}
            return image, target

        class_label = int(labels[0][0])
        x1, y1, x2, y2 = map(float, labels[0][1:])

        box = [x1, y1, x2, y2]

        box = torch.as_tensor(box, dtype=torch.float32)
        class_label = torch.as_tensor(class_label, dtype=torch.int64)

        target = {"box": box, "class_label": class_label}

        return image, target


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)


def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


dataset = MedicalImageDataset(image_dir=IMAGES, label_dir=LABELS, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
a, b = random_split(dataset, [train_size, test_size])


trainset = DataLoader(a, batch_size=4, shuffle=True)
testset = DataLoader(b, batch_size=4, shuffle=False)


if __name__ == "__main__":
    for data in trainset:
        print(data)
        break

    import matplotlib.pyplot as plt

    print(data[0][0].shape)
    plt.imshow(data[0][0].permute(1, 2, 0))
    plt.show()

    # for images, targets in trainset:
    #     print(f"Images batch shape: {len(images)}, {images[0].shape}")
    #     print(f"Targets batch: {targets}")
    #     break
