import cv2 as cv
import matplotlib.pyplot as plt
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

