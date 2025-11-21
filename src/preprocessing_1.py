import cv2 as cv
import numpy as np
import os
import shutil


def process_image(img: np.ndarray,
                  blur_kernel : tuple[int, int] = (3, 3),
                  gamma: float = 0.5,
                  blend_weights: tuple[float, float] = (-0.4, 1.6)) -> np.ndarray:
    # Apply Gaussian blur
    img = cv.GaussianBlur(img, blur_kernel, sigmaX=0)
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i/255.0)**inv_gamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv.LUT(img, table)
    # Blend images to enhance contrast
    img = cv.addWeighted(img, blend_weights[0], img_gamma, blend_weights[1], 0)
    return img


def preprocess_1() -> None:
    for dirpath, _, filenames in os.walk("data"):
        # Transform and save images
        if dirpath.endswith("images"):
            for filename in filenames:
                # Read image in grayscale
                img = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                # Process image
                img = process_image(img)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                # Save preprocessed image
                out_dir = dirpath.replace("data", "data_preprocessed")
                os.makedirs(out_dir, exist_ok=True)
                base, _ = os.path.splitext(filename)
                out_path = os.path.join(out_dir, base + ".jpg")
                img_to_save = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
                cv.imwrite(out_path, img_to_save, [int(cv.IMWRITE_JPEG_QUALITY), 95])
        # Copy label files
        elif dirpath.endswith("labels"):
            shutil.copytree(dirpath, dirpath.replace("data", "data_preprocessed"), dirs_exist_ok=True)
        # Pass if other directories
        else:
            continue
    # Copy data.yaml
    shutil.copy("data/data.yaml", "data_preprocessed/data.yaml")

if __name__ == "__main__":
    preprocess_1()