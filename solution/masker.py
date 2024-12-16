import cv2 # wenn ihr das package nicht habt, führt pip install opencv-python aus
from pathlib import Path
import numpy as np

def create_part_mask(part_image_path: Path, invert_mask: bool = False, blur_method = "median", adaptive = True, area_filter = True, show_images = False) -> np.ndarray:
    # Load the image
    img = cv2.imread(str(part_image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read the image from {part_image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    if blur_method == "gaussian":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif blur_method == "median":
        gray = cv2.medianBlur(gray, 5)

    if adaptive:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 5)
    else:
        # Otsu
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if area_filter:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        area_threshold = 100
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_threshold:
                cv2.drawContours(clean_mask, [cnt], -1, (255, 255, 255), -1)
        mask = clean_mask

    if(show_images):
        cv2.imshow("Before", img)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    masks_dir = (Path(__file__).resolve().parent / "_masks")
    masks_dir.mkdir(exist_ok=True)

    original_stem = part_image_path.stem
    mask_filename = masks_dir / f"{original_stem}_mask.png"
    original_filename = masks_dir / f"{original_stem}_original.png"

    cv2.imwrite(str(original_filename), img)
    cv2.imwrite(str(mask_filename), mask)

    return mask

