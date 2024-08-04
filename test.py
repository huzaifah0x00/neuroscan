import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data import testset
from model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_DIMENSIONS = 139, 132  # width, height


def load_model(path):
    model = Net().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def yolo_to_pixel(box):
    img_width, img_height = IMAGE_DIMENSIONS
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


def visualize_boxes(image, true_box, pred_box, iou):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap="gray")

    true_rect = patches.Rectangle(
        (true_box[0], true_box[1]),
        true_box[2] - true_box[0],
        true_box[3] - true_box[1],
        linewidth=2,
        edgecolor="g",
        facecolor="none",
    )
    pred_rect = patches.Rectangle(
        (pred_box[0], pred_box[1]),
        pred_box[2] - pred_box[0],
        pred_box[3] - pred_box[1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )

    ax.add_patch(true_rect)
    ax.add_patch(pred_rect)

    plt.text(10, 10, f"IoU: {iou:.2f}", fontsize=12, color="black", bbox=dict(facecolor="white", alpha=0.5))

    plt.show()


def test_model(model, test_loader, visualize=False):
    total_iou, count_iou = 0, 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            X, y = data
            X = X.to(device)
            outputs = model(X)

            predicted_boxes = outputs[:, :-4]
            true_boxes = y["box"].to(device)

            for idx in range(true_boxes.size(0)):
                pred_box = yolo_to_pixel(predicted_boxes[idx].tolist())
                true_box = yolo_to_pixel(true_boxes[idx].tolist())

                iou = compute_iou(pred_box, true_box)
                total_iou += iou
                count_iou += 1

                if visualize:
                    visualize_boxes(X[idx], true_box, pred_box, iou)

    print(f"Average IoU: {total_iou / count_iou}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the actual and predicted box.")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    model = Net().to(device)
    model.load_state_dict(torch.load("trained_model_state_dict.pytorch", weights_only=True))
    model.eval()

    test_model(model, testset, visualize=args.visualize)
