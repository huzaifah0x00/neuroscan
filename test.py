import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from data import testset
from main import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = Net().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def compute_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def test_model(model, test_loader):
    total_iou, count_iou = 0, 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            X, y = data
            X = X.to(device)
            outputs = model(X)
            
            predicted_boxes = outputs[:, :-4]
            true_boxes = y["box"].to(device)
            
            for idx in range(true_boxes.size(0)):
                iou = compute_iou(predicted_boxes[idx].tolist(), true_boxes[idx].tolist())
                total_iou += iou
                count_iou += 1

    print(f'Average IoU: {total_iou / count_iou}')


model = Net().to(device)
model.load_state_dict(torch.load("trained_model_state_dict.pytorch", weights_only=True))
model.eval()

test_model(model, testset)
