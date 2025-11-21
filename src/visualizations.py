import cv2 as cv
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils import get_random_image_id, get_original_image_path, get_processed_image_path



def compare_original_and_processed(root_dir: str) -> None:
    """Display original and processed images side by side for comparison."""
    # Get a random image
    image_id, dataset = get_random_image_id(root_dir)
    original_path = get_original_image_path(image_id, dataset)
    processed_path = get_processed_image_path(image_id, dataset)
    # Read images
    image1 = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(processed_path, cv.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        raise FileNotFoundError(f"Could not read image(s): {original_path}, {processed_path}")

    # Plot images side by side
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def compare_with_histograms(root_dir: str):
    """Top row: original / processed images. Bottom row: their color histograms (R,G,B) as density (normalized)."""
    image_id, dataset = get_random_image_id(root_dir)
    original_path = get_original_image_path(image_id, dataset)
    processed_path = get_processed_image_path(image_id, dataset)

    image1 = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(processed_path, cv.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        raise FileNotFoundError(f"Could not read image(s): {original_path}, {processed_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top row: images
    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title("Processed Image")
    axes[0, 1].axis("off")

    # Bottom row: color histograms (density)

    axes[1, 0].hist(image1.ravel(), bins=64, density=True, color='dimgray')
    axes[1, 1].hist(image2.ravel(), bins=64, density=True, color='dimgray')

    axes[1, 0].set_xlim(0, 255)
    axes[1, 1].set_xlim(0, 255)
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_image_with_bboxes(root_dir: str, original_image: bool = False) -> None:
    """Display original and processed images with bounding boxes and class labels."""
    import cv2 as cv
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Get a random image
    image_id, dataset = get_random_image_id(root_dir)
    if original_image:
        image_path = get_original_image_path(image_id, dataset)
    else:
        image_path = get_processed_image_path(image_id, dataset)

    # Read image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Bounding boxes file
    bboxes_path = image_path.replace("images", "labels")
    bboxes_path = os.path.splitext(bboxes_path)[0] + ".txt"

    bboxes_labels = np.loadtxt(bboxes_path)
    if bboxes_labels.ndim == 1:
        bboxes_labels = np.expand_dims(bboxes_labels, axis=0)

    h, w = image.shape

    # Plot image & bboxes
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")

    for bbox in bboxes_labels:
        cls, x_center, y_center, box_width, box_height = bbox

        x_min = int((x_center - box_width / 2) * w)
        y_min = int((y_center - box_height / 2) * h)
        bw = int(box_width * w)
        bh = int(box_height * h)

        # Draw rectangle (green)
        rect = plt.Rectangle(
            (x_min, y_min), bw, bh,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        plt.gca().add_patch(rect)

        # Add class label
        plt.text(
            x_min, y_min - 5,
            f"class {int(cls)}",
            color='lime',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
        )

    plt.title(f"Image with marked bounding boxes")
    plt.axis("off")
    plt.show()

