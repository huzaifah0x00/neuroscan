import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from data import trainset
from model import Net
from test import compute_iou, yolo_to_pixel

device = torch.device("cuda:0")


def compute_loss(predictions, targets):
    predicted_boxes = predictions[:, :-4]
    # predicted_classes = predictions[:, -4:]
    true_boxes = targets["box"].to(device)
    # true_classes = targets["class_label"].to(device)

    loss_bbox = bbox_loss_function(predicted_boxes, true_boxes)
    # loss_class = class_loss_function(predicted_classes, true_classes)

    # iou = compute_iou(yolo_to_pixel(predicted_boxes[0].tolist()), yolo_to_pixel(true_boxes[0].tolist()))
    # print("iou", iou, "loss", loss_bbox)

    # total_loss = loss_bbox + loss_class
    total_loss = loss_bbox
    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or retrain the model.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model from scratch")
    args = parser.parse_args()

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bbox_loss_function = torch.nn.SmoothL1Loss()
    class_loss_function = torch.nn.CrossEntropyLoss()

    if not args.retrain:
        try:
            model.load_state_dict(torch.load("trained_model_state_dict.pytorch", weights_only=True))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Training from scratch.")

    EPOCHS = 10
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
