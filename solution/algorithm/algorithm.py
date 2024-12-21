from PIL import Image
import cv2
from matplotlib import patches
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def run_algorithm(part_path, gripper_path):
    """
    Runs the algorithm to find the best position and angle for the gripper on the part.
    """
    part_img = Image.open(part_path).convert("L")
    gripper_img = Image.open(gripper_path).convert("RGBA")

    part_img = reduce_image_quality(part_img, 0.2)
    gripper_img = reduce_image_quality(gripper_img, 0.2)

    part_mask = np.array(part_img) > 0
    gripper_mask = np.array(gripper_img)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden

    # Calculate the best position and angle for the gripper on the part
    result = calc_best_position(part_mask, gripper_mask)
 
    if result is not None:
        best_x, best_y, best_angle = result
    else: 
        return None

    # Visualize the gripper on the part image
    visualize_gripper_on_part(part_mask, gripper_mask, best_x, best_y, best_angle)


    print(f"Best Position: x={best_x}, y={best_y}, angle={best_angle}")
    return best_x, best_y, best_angle

def reduce_image_quality(image, scale_factor):
    """
    Reduces the image quality by scaling it down using PIL.
    """

    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    downscaled_image = image.resize((new_width, new_height), Image.NEAREST)

    return downscaled_image


def calc_best_position(part_mask, gripper_mask):
    """
    Calculates the best position and angle for the gripper on the part using template matching.
    """
    for x in range(0, part_mask.shape[1], 2):
        for y in range(0, part_mask.shape[0], 2):
            for angle in range(0, 360, 8):

                if is_valid_configuration(part_mask, gripper_mask, x, y, angle):
                    print("Valid configuration found.")
                    print(f"Position: x={x}, y={y}, angle={angle}")
                    return x, y, angle

    return None



def is_valid_configuration(part_mask, gripper_mask, x, y, angle):
    """
    Check if the gripper is in a valid configuration on the part.
    Returns True if the gripper is fully inside the part, otherwise False.
    """


    rotated_gripper_mask = rotate(gripper_mask, -angle , reshape=True, order=0)
    rotated_gripper_mask = ~rotated_gripper_mask

    rotated_gripper_height, rotated_gripper_width = rotated_gripper_mask.shape

    
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
    for i in range(gripper_left, gripper_right):
        for j in range(gripper_top, gripper_bottom):
            
            if 0 <= i < part_mask.shape[1] and 0 <= j < part_mask.shape[0]:  # Check if the pixel is inside the part image
                if not rotated_gripper_mask[j - gripper_top, i - gripper_left] and part_mask[j, i]:
                    return False  # no valid configuration
            
            else:              
                return False
    
    print(x, y, angle)
    print("Valid configuration found.")
    
    return True
    #return True  # The gripper is fully inside the part



def visualize_gripper_on_part(part_mask, gripper_mask, x, y, angle):
    """
    Visualizes the gripper on the part image at the given position and angle.
    """

    
    
    # Rotate the gripper image
    rotated_gripper_mask = rotate(gripper_mask, angle, reshape=True, order=0)

    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(part_mask, cmap="gray", origin="upper")

    
    gripper_height, gripper_width = rotated_gripper_mask.shape

    # Add the gripper image to the plot
    ax.imshow(rotated_gripper_mask, cmap="Blues", extent=(x - (gripper_width / 2), x + (gripper_width / 2), y - (gripper_height / 2), y + (gripper_height / 2)), alpha=0.3, origin="upper")

    
    ax.set_title("Gripper Visualisierung")
    plt.gca().invert_yaxis()
    plt.show()

def findPartsMinimalRadius(image):
    width, height = image.size
    # - 1 because when one subtracted the gripper would be always invalid while places near an edge
    return min(width, height) - 1

