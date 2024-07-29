import re
import os
import tkinter as tk
from tkinter import Label, Canvas
from PIL import Image, ImageTk
import json
from pprint import pprint


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


image_dir = "dataset/images"
label_dir = "dataset/labels"


image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")], key=natural_sort_key)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")], key=natural_sort_key)

print(f"total image files: {len(image_files)}")
print(f"total label files: {len(label_files)}")

pprint(image_files[0:10])
pprint(label_files[0:10])

index = 0


def load_image(index):
    image_path = os.path.join(image_dir, image_files[index])
    label_path = os.path.join(label_dir, label_files[index])

    image = Image.open(image_path)
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f]

    return image, labels


def display_image(canvas, image, labels):
    canvas.delete("all")
    img_width, img_height = image.size

    scale_w = 250 / img_width
    scale_h = 250 / img_height
    scale = min(scale_w, scale_h)

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image)

    x_offset = (250 - new_width) // 2
    y_offset = (250 - new_height) // 2
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=image_tk)
    canvas.image_tk = image_tk

    for label in labels:
        center_x, center_y, bbox_width, bbox_height = map(float, label[1:])
        bbox_x = x_offset + (center_x - bbox_width / 2) * new_width
        bbox_y = y_offset + (center_y - bbox_height / 2) * new_height
        bbox_w = bbox_width * new_width
        bbox_h = bbox_height * new_height
        canvas.create_rectangle(bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h, outline="red", width=2)


def show_next_image(event=None):
    global index
    index = (index + 1) % len(image_files)
    image, labels = load_image(index)
    print(f"Showing image: {image_files[index]}")
    display_image(canvas, image, labels)


def show_prev_image(event=None):
    global index
    index = (index - 1) % len(image_files)
    image, labels = load_image(index)
    display_image(canvas, image, labels)


root = tk.Tk()
root.title("Image Browser with Bounding Boxes")
root.bind("<Right>", show_next_image)
root.bind("<Left>", show_prev_image)

canvas = Canvas(root, width=250, height=250)
canvas.pack()

image, labels = load_image(index)
display_image(canvas, image, labels)

root.mainloop()
