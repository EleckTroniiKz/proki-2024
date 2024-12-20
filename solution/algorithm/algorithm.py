from PIL import Image
import cv2
from matplotlib import patches
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def reduce_image_quality(image, scale_factor):
    """
    Reduces the image quality by scaling it down using PIL.
    """

    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    downscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    return downscaled_image


def create_mask_from_png(part_img):
    """
    Creates a binary mask from a PNG image.
    """
    threshold = 1  
    mask = np.array(part_img) < threshold


    return mask


def is_valid_configuration(part_img, gripper_img, x, y, angle):
    """
    Check if the gripper is in a valid configuration on the part.
    Returns True if the gripper is fully inside the part, otherwise False.
    """

    # Load the part image and create a binary mask
    part_mask = create_mask_from_png(part_img)
    
    # Load the gripper image and rotate it
    rotated_gripper = gripper_img.rotate(angle, expand=True)
    #rotated_gripper_width, rotated_gripper_height = gripper_img.size
    

    # Create a binary mask for the rotated gripper
    gripper_mask = np.array(rotated_gripper)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden
    rotated_gripper_height, rotated_gripper_width = gripper_mask.shape

    # Calculate the bounding box of the gripper
    gripper_left = x - rotated_gripper_width // 2
    gripper_right = x + rotated_gripper_width // 2
    gripper_top = y - rotated_gripper_height // 2
    gripper_bottom = y + rotated_gripper_height // 2


    if rotated_gripper_height % 2 == 0:
        gripper_bottom -= 1

    if rotated_gripper_width % 2 == 0:
        gripper_right -= 1

    # oder (ToDo: Test for both)
    #if rotated_gripper_height % 2 == 0:
    #    gripper_top += 1
    #if rotated_gripper_width % 2 == 0:
    #    gripper_left += 1


    # Check if the gripper is fully inside the part
    for i in range(gripper_top, gripper_bottom):
        for j in range(gripper_left, gripper_right):
            if 0 < i < part_mask.shape[0] and 0 < j < part_mask.shape[1]:  # Check if the pixel is inside the part image
                if gripper_mask[i - gripper_top, j - gripper_left] and not part_mask[i, j]:                    
                    return False  # no valid configuration
            else:
                return False
    return True  # The gripper is fully inside the part


def calc_best_position(part_image, gripper_image):
    """
    Calculates the best position and angle for the gripper on the part using template matching.
    """
    for x in range(0, part_image.width, 1):
        for y in range(0, part_image.height, 13):
            for angle in range(0, 360, 45):
                #print(f"Checking position: x={x}, y={y}, angle={angle}")  
                if is_valid_configuration(part_image, gripper_image, x, y, angle):
                    print("Valid configuration found.")
                    print(f"Position: x={x}, y={y}, angle={angle}")
                    return x, y, angle

    return None




def visualize_gripper_on_part(part_img, gripper_img, x, y, angle):
    """
    Visualizes the gripper on the part image at the given position and angle.
    """

    
    
    # Rotate the gripper image
    rotated_gripper = gripper_img.rotate(angle, expand=True)

    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(part_img, cmap="gray", origin="upper")

    
    grip_width, grip_height = rotated_gripper.size

    # Add the gripper image to the plot
    ax.imshow(rotated_gripper, extent=(x - (grip_width / 2), x + (grip_width / 2), y - (grip_height / 2), y + (grip_height / 2)) , alpha=1)

    
    ax.set_title("Gripper Visualisierung")
    plt.gca().invert_yaxis()
    plt.show()



def run_algorithm(part_path, gripper_path):
    """
    Runs the algorithm to find the best position and angle for the gripper on the part.
    """
    part_img = Image.open(part_path).convert("L")
    gripper_img = Image.open(gripper_path).convert("RGBA")

    part_img = reduce_image_quality(part_img, 0.5)
    gripper_img = reduce_image_quality(gripper_img, 0.5)

    # Calculate the best position and angle for the gripper on the part
    result = calc_best_position(part_img, gripper_img)

    visualize_gripper_on_part(part_img, gripper_img, 95, 55, 0)
    
    if result is not None:
        best_x, best_y, best_angle = result

    else: 
        return None

    # Visualize the gripper on the part image
    visualize_gripper_on_part(part_img, gripper_img, best_x, best_y, best_angle)


    print(f"Best Position: x={best_x}, y={best_y}, angle={best_angle}")
    return best_x, best_y, best_angle