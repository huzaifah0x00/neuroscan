import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from data import trainset

device = torch.device("cuda:0")

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(17408, 512),
            nn.ReLU(),
            nn.Linear(512, 4 + 4),
        )

    def forward(self, x):
        x = self.base_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def compute_loss(predictions, targets):
    predicted_boxes = predictions[:, :-4]
    predicted_classes = predictions[:, -4:]
    true_boxes = targets["box"].to(device)
    true_classes = targets["class_label"].to(device)
    loss_bbox = bbox_loss_function(predicted_boxes, true_boxes)
    loss_class = class_loss_function(predicted_classes, true_classes)
    total_loss = loss_bbox + loss_class
    return total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or retrain the model.')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model from scratch')
    args = parser.parse_args()

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bbox_loss_function = torch.nn.SmoothL1Loss()
    class_loss_function = torch.nn.CrossEntropyLoss()

    if not args.retrain:
        try:
            model.load_state_dict(torch.load("trained_model_state_dict.pytorch"))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Training from scratch.")

    EPOCHS = 5
    for epoch in range(EPOCHS):
        for data in tqdm(trainset):
            X, y = data
            X = X.to(device)
            model.zero_grad()
            outputs = model(X)
            loss = compute_loss(outputs, y)
            loss.backward()
            optimizer.step()
        print(loss)

    torch.save(model.state_dict(), "trained_model_state_dict.pytorch")
