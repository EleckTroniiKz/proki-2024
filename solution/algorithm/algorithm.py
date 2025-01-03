from collections import deque
from PIL import Image
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from masker import create_part_mask


def run_algorithm(part_path, gripper_path):
    """
    Runs the algorithm to find the best position and angle for the gripper on the part.
    """
    part_img = create_part_mask(part_path).convert("L")
    gripper_img = Image.open(gripper_path).convert("RGBA")

    # Calculate the best position and angle for the gripper on the part
    #result = calc_best_position(part_img, gripper_img)

    result  = calc_best_position(part_img, gripper_img)

    if result is not None:
        best_x, best_y, best_angle = result
    else: 
        return None

    # Visualize the gripper on the part image
    
    visualize_gripper_on_part(part_img, gripper_img, best_x, best_y, best_angle)


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



def is_valid_configuration(part_mask, gripper_mask, x, y, angle):
    """
    Check if the gripper is in a valid configuration on the part.
    Returns True if the gripper is fully inside the part, otherwise False.
    """

    #+, true, means white so a hole
    #-, false means solid part
    rotated_gripper_mask = rotate(gripper_mask, -angle , reshape=True, order=0)
    rotated_gripper_mask = ~rotated_gripper_mask # False if the pixel is part of the gripper (marked black), True otherwise
    
    rotated_gripper_height, rotated_gripper_width = rotated_gripper_mask.shape
    
    # Calculate the bounding box of the gripper
    gripper_left = x - rotated_gripper_width // 2
    gripper_right = x + rotated_gripper_width // 2
    gripper_top = y - rotated_gripper_height // 2
    gripper_bottom = y + rotated_gripper_height // 2

    # Adjust the bounding box if the gripper has an even width or height
    if rotated_gripper_height % 2 == 0:
        gripper_bottom -= 1

    if rotated_gripper_width % 2 == 0:
        gripper_right -= 1


    # Check if the gripper is fully inside the part; Note: range(2,5) is for [2,3,4]
    for i in range(gripper_left, gripper_right+1):
        for j in range(gripper_top, gripper_bottom+1):
                      
            if 0 <= i < part_mask.shape[1] and 0 <= j < part_mask.shape[0]:  # Check if the pixel is inside the part image
                if not rotated_gripper_mask[j - gripper_top, i - gripper_left] and part_mask[j, i]:  # Check if the gripper is on the part   
                    return False  # gripper is on hole
            
            else:              
                return False # gripper not fully inside the part shape
 
    return True  # The gripper is fully inside the part



def visualize_gripper_on_part(part_img, gripper_img, x, y, angle):
    """
    Visualizes the gripper on the part image at the given position and angle.
    """
    part_mask = np.array(part_img) > 0
    # True if pixel is part of the gripper, False otherwise
    gripper_mask = np.array(gripper_img)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden
    
    # Rotate the gripper image
    rotated_gripper_mask = rotate(gripper_mask, angle, reshape=True, order=0)

    # Create a new figure
    fig, ax = plt.subplots()
    ax.imshow(part_mask, cmap="gray", origin="upper")
    
    rotated_gripper_height, rotated_gripper_width = rotated_gripper_mask.shape

    # Calculate the bounding box of the gripper
    gripper_left = x - rotated_gripper_width // 2 
    gripper_right = x + rotated_gripper_width // 2 
    gripper_top = y - rotated_gripper_height // 2
    gripper_bottom = y + rotated_gripper_height // 2

    gripper_left = gripper_left - 0.5
    gripper_top = gripper_top - 0.5

    if rotated_gripper_height % 2 == 0:
        gripper_bottom -= 0.5
    else:
        gripper_bottom += 0.5

    if rotated_gripper_width % 2 == 0:
        gripper_right -= 0.5
    else:
        gripper_right += 0.5


    # Define the bounding box of the gripper
    gripper_on_part = (gripper_left, gripper_right, gripper_top, gripper_bottom)
    # Add the gripper image to the plot
    ax.imshow(rotated_gripper_mask, cmap="Blues", extent = gripper_on_part, alpha=0.3, origin="upper")

    # # ax = mark_points_on_figure(ax, get_middle_point_of(part_mask), (x,y))
    # midPoint = get_middle_point_of(part_mask)
    
    # # dye middel point of gripper blue
    # if rotated_gripper_mask.shape[0] % 2 == 0:  # Even widht
    #     for j in range(part_mask.shape[0]):
    #         ax.plot(midPoint[0], j, marker='o', color='#FF0000')
    # else:
    #     ax.plot(midPoint[0], midPoint[1], marker='o', color='#FF0000')

    # if part_mask.shape[1] % 2 == 0:  # Even height
    #     for i in range(part_mask.shape[1]/ 2, part_mask.shape[1]//2 + 1):
    #         ax.plot(i, midPoint[1], marker='o', color='#0000FF')
    # else:
    #     ax.plot(midPoint[0], midPoint[1], marker='o', color='#0000FF')
    
    
    ax.plot(x, y, marker='o', color='#0000FF')


    ax.set_title("Gripper Visualisierung")
    plt.gca().invert_yaxis()
    plt.show()

def mark_points_on_figure(ax, point1, point2):
        """
        Marks two given points on the part image using the provided plot axis.

        Args:
            ax (matplotlib.axes.Axes): The plot axis to draw on.
            part_mask (numpy.ndarray): The part image mask.
            point1 (tuple): The (x, y) coordinates of the first point.
            point2 (tuple): The (x, y) coordinates of the second point.
        """
        # ax.imshow(part_mask, cmap="gray", origin="upper")

        # Mark the first point
        ax.plot(point1[0], point1[1], 'ro')  # 'ro' means red color, circle marker

        # Mark the second point
        ax.plot(point2[0], point2[1], 'bo')  # 'bo' means blue color, circle marker

        # ax.set_title("Marked Points on Part")
        # plt.gca().invert_yaxis()


def findPartsMinimalRadius(image):
    """
    Calculate the minimal radius for parts to determine which pixels can be blacklisted 
    without even looking at them because they will always be to close to an edge.
    """
    width, height = image.size
    # - 1 because when one subtracted the gripper would be always invalid while places near an edge
    return min(width, height) - 1

def get_middle_point_of(part_mask) -> tuple:
    """
    Calculates the middle point of an array.
    """
    # TODO(Torben) check here for odd middle of part width and/or height and think about how to handle
    height, width = part_mask.shape
    # if (width % 2 == 1): 
    middle_x = width // 2 + 1
    # else : 
    #     middle_x = (width // 2, width // 2 + 1)
    
    # if (height % 2 == 1): 
    middle_y = height // 2 + 1
    # else : 
    #     middle_y = (height // 2, height // 2 + 1)
    return middle_x, middle_y


def calc_best_position(part_img, gripper_img):
    """
    Calculates the best position and angle for the gripper on the part using a breadth-first search
    """

    
    part_mask = np.array(part_img) > 0
    # True if pixel is part of the gripper, False otherwise
    gripper_mask = np.array(gripper_img)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden

    scale_factor = 0.2

    reduced_part_img = reduce_image_quality(part_img, scale_factor)
    reduced_gripper_img = reduce_image_quality(gripper_img, scale_factor)

    reduced_part_mask = np.array(reduced_part_img) > 0
    # True if pixel is part of the gripper, False otherwise
    reduced_gripper_mask = np.array(reduced_gripper_img)[:, :, 3] > 0  # Nur Alpha-Kanal verwenden

    # Calculate the middle point of the part mask
    midPoint = get_middle_point_of(reduced_part_mask)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([midPoint])
    visited = set()
    visited.add(midPoint)

    while queue:
        x, y = queue.popleft()
        for angle in range(0, 360, 1):
            if (count_overlapping_pixels(reduced_part_mask, reduced_gripper_mask, x, y, angle) < 3):
                check_x = int(x // scale_factor)
                check_y = int(y // scale_factor)
                #visualize_gripper_on_part(reduced_part_img, reduced_gripper_img, x, y, angle)
                if is_valid_configuration(part_mask, gripper_mask, check_x, check_y, angle):
                    return check_x, check_y, angle


        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < reduced_part_mask.shape[1] and 0 <= ny < reduced_part_mask.shape[0] and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return None  # If no valid position is found



def count_overlapping_pixels(reduced_part_mask, reduced_gripper_mask, x, y, angle):
    """
    Counts the number of overlapping pixels between the gripper and the part at the given position and angle.
    """

    #+, true, means white so a hole
    #-, false means solid part
    rotated_gripper_mask = rotate(reduced_gripper_mask, -angle , reshape=True, order=0)
    rotated_gripper_mask = ~rotated_gripper_mask # False if the pixel is part of the gripper (marked black), True otherwise
    
    rotated_gripper_height, rotated_gripper_width = rotated_gripper_mask.shape
    
    # Calculate the bounding box of the gripper
    gripper_left = x - rotated_gripper_width // 2
    gripper_right = x + rotated_gripper_width // 2
    gripper_top = y - rotated_gripper_height // 2
    gripper_bottom = y + rotated_gripper_height // 2

    # Adjust the bounding box if the gripper has an even width or height
    if rotated_gripper_height % 2 == 0:
        gripper_bottom -= 1

    if rotated_gripper_width % 2 == 0:
        gripper_right -= 1
    
    
    counter = 0

    # Check if the gripper is fully inside the part; Note: range(2,5) is for [2,3,4]
    for i in range(gripper_left, gripper_right+1):
        for j in range(gripper_top, gripper_bottom+1):
                      
            if 0 <= i < reduced_part_mask.shape[1] and 0 <= j < reduced_part_mask.shape[0]:  # Check if the pixel is inside the part image
                if not rotated_gripper_mask[j - gripper_top, i - gripper_left] and reduced_part_mask[j, i]:  # Check if the gripper is on the part   
                    counter += 1  
            else:              
                counter += 1
    return counter  



