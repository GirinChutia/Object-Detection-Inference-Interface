import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names


def plot_boxes_on_image(results, image_path, class_names, output_path="output.jpg"):
    image = cv2.imread(image_path)

    boxes = results["boxes"]
    scores = results["scores"]
    class_indices = results["classes"]

    for box, score, class_index in zip(boxes, scores, class_indices):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)

        label = f"{class_names[int(class_index)]}: {float(score):.2f}"
        box_color = (0, 0, 255)
        label_bg_color = (255, 255, 255)
        label_text_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_thickness = 2
        label_bg_offset = 10

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, line_thickness)

        label_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        label_bg_x1 = x1
        label_bg_y1 = y1 - label_bg_offset - label_size[1]
        label_bg_x2 = x1 + label_size[0]
        label_bg_y2 = y1 - label_bg_offset
        cv2.rectangle(
            image,
            (label_bg_x1, label_bg_y1),
            (label_bg_x2, label_bg_y2),
            label_bg_color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - label_bg_offset),
            font,
            font_scale,
            label_text_color,
            1,
        )

    cv2.imwrite(output_path, image)


def plot_results(
    results,
    image_path,
    class_names,
    figsize=(10, 10),
    save=False,
    save_path="output.jpg",
):
    image = cv2.imread(image_path)
    boxes = results["boxes"]
    scores = results["scores"]
    class_indices = results["classes"]
    colors = _COLORS

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for box, score, class_index in zip(boxes, scores, class_indices):
        x1, y1, x2, y2 = box
        classname = class_names[class_index]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        color = colors[class_index]  # Select color based on class_index
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x + 2,
            y - 5,
            f"{classname} {score:.2f}",
            fontsize=10,
            color=color,
            bbox=dict(facecolor="white", alpha=0.25, edgecolor=color),
        )

    plt.axis("off")
    if save:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
