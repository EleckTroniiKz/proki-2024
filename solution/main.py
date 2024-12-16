import cv2 # wenn ihr das package nicht habt, führt pip install opencv-python aus
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

from rich.progress import track
import pandas as pd

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

    return mask


def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    print(part_image_path)
    coords = findCenterOfGripper(part_image_path, gripper_image_path)
    if coords is None:
        print(f"Kein Schwerpunkt für {part_image_path}. Zeile überspringen.")
        return None, None, None


    return coords
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """

    a = create_part_mask(part_image_path)

    return 100.1, 95, 91.2


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
