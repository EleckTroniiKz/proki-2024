import cv2
import numpy as np
from PIL import Image
import os
from PIL import Image
# Load the image
# Load the first provided part image for processing

def create_part_mask(part_image_path):
    part_image_blur = cv2.imread(part_image_path)
    blurred_image = cv2.GaussianBlur(part_image_blur, (5, 5), 0)
    part_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(part_image)

    tile_size = 100
    h, w = equalized_image.shape
    result_mask = np.zeros_like(equalized_image)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = equalized_image[y:y+tile_size, x:x+tile_size]

            tile_thresh = cv2.adaptiveThreshold(
                tile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            result_mask[y:y+tile_thresh.shape[0], x:x+tile_thresh.shape[1]] = tile_thresh

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
    filtered_mask = np.zeros_like(result_mask)

    min_area = 100  # Minimum area to retain a component
    for i in range(1, num_labels):  # Skip the background label
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    output_dir = "_masks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join("_masks", f"{os.path.basename(part_image_path)}_output.png")
    print(part_image_path)

    cv2.imwrite(output_path, filtered_mask)

    return Image.fromarray(filtered_mask)
