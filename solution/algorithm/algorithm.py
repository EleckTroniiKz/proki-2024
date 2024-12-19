from PIL import Image
import cv2
from matplotlib import patches
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cairosvg


# def calcCenterOfGripper(teil_img, greifer_img):
    
#     teil_maske = np.array(teil_img) < 140
#     greifer_maske = np.array(greifer_img) < 10

#     plt.imshow(teil_maske, cmap='gray')
#     plt.title('Binary Mask')
#     plt.show()

#     # Mittelpunkt des Teils berechnen
#     # shape = Dimensionen des Bildes
#     h, w = teil_maske.shape
#     x_part, y_part = w // 2, h // 2

#     beste_distanz = float("inf")
#     beste_position = None

#     # Raster-Suche
#     for y in range(0, h, 50):
#         for x in range(0, h, 50):
#             for winkel in range(0, 360, 45):
#                 rotate(greifer_maske, 45)
#                 if 1:
#                     distanz = (x - x_part)**2 + (y - y_part)**2
#                     if distanz < beste_distanz:
#                         beste_distanz = distanz
#                         beste_position = (x, y, winkel)
#     print(beste_position)
#     return beste_position
            


def create_mask_from_png(image_path):
    """
    Creates a binary mask from a PNG image.
    """
    img = Image.open(image_path).convert("L")  
    threshold = 1  
    mask = np.array(img) < threshold


    return mask


def is_valid_configuration(part_image_path, gripper_image_path, x, y, angle):
    """
    Check if the gripper is in a valid configuration on the part.
    Returns True if the gripper is fully inside the part, otherwise False.
    """
    # Load the part image and create a binary mask
    part_mask = create_mask_from_png(part_image_path)
    
    # Load the gripper image and rotate it
    gripper_img = Image.open(gripper_image_path).convert("RGBA")
    grip_width, grip_height = gripper_img.size
    rotated_gripper = gripper_img.rotate(angle, resample=Image.BICUBIC, center=(grip_width / 2, grip_height / 2))
    
    # Create a binary mask for the rotated gripper
    gripper_mask = np.array(rotated_gripper)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden
    
    # Calculate the bounding box of the gripper
    gripper_left = x - grip_width // 2
    gripper_right = x + grip_width // 2
    gripper_top = y - grip_height // 2
    gripper_bottom = y + grip_height // 2

    # Check if the gripper is fully inside the part
    for i in range(gripper_left, gripper_right):
        for j in range(gripper_top, gripper_bottom):
            if 0 <= i < part_mask.shape[1] and 0 <= j < part_mask.shape[0]:  # Nur gültige Indizes
                if gripper_mask[j - gripper_top, i - gripper_left] and not part_mask[j, i]:
                    return False  # no valid configuration
    return True  # The gripper is fully inside the part


def calc_best_position(part_image, gripper_image):
    """
    Calculates the best position and angle for the gripper on the part using template matching.
    """
    
    # Convert images to grayscale
    part_gray = np.array(part_image.convert("L"))
    gripper_gray = np.array(gripper_image.convert("L"))
    
    
    grip_width, grip_height = gripper_image.size

    best_score = -1  # Schlechte Übereinstimmung initialisieren
    best_x, best_y, best_angle = 0, 0, 0  # Initiale Werte für die beste Position

    for angle in range(0, 360, 10):  

        rotated_gripper = gripper_image.rotate(angle, resample=Image.BICUBIC, center=(grip_width / 2, grip_height / 2))

        
        rotated_gripper_gray = np.array(rotated_gripper.convert("L"))

        # Template Matching
        result = cv2.matchTemplate(part_gray, rotated_gripper_gray, method=cv2.TM_CCOEFF_NORMED)

        # Search for the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if the current match is better than the previous best match
        if max_val > best_score:
            best_score = max_val
            best_x, best_y = max_loc
            best_angle = angle

    return 130, 173, 0




def visualize_gripper_on_part(part_path, gripper_path, x, y, angle):
    """
    Visualizes the gripper on the part image at the given position and angle.
    """

    part_img = Image.open(part_path).convert("L")
    gripper_img = Image.open(gripper_path).convert("RGBA")
    
    # Rotate the gripper image
    rotated_gripper = gripper_img.rotate(angle, resample=Image.BICUBIC, center=(gripper_img.width / 2, gripper_img.height / 2))

    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(part_img, cmap="gray", origin="upper")

    
    grip_width, grip_height = gripper_img.size

    # Add the gripper image to the plot
    ax.imshow(gripper_img, extent=(x - (grip_width / 2), x + (grip_width / 2), y - (grip_height / 2), y + (grip_height / 2)) , alpha=1)

    
    ax.set_title("Gripper Visualisierung")
    plt.gca().invert_yaxis()
    plt.show()



def run_algorithm(part_path, gripper_path):
    """
    Runs the algorithm to find the best position and angle for the gripper on the part.
    """
    part_img = Image.open(part_path)
    gripper_img = Image.open(gripper_path)

    # Calculate the best position and angle for the gripper on the part
    best_x, best_y, best_angle = calc_best_position(part_img, gripper_img)

    # Visualize the gripper on the part image
    visualize_gripper_on_part(part_path, gripper_path, best_x, best_y, best_angle)
    print(is_valid_configuration(part_path, gripper_path, best_x, best_y, best_angle))


    print(f"Best Position: x={best_x}, y={best_y}, angle={best_angle}")
    return best_x, best_y, best_angle