import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import numpy as np
from scipy.ndimage import rotate

# # Define the U-Net model
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# # Custom dataset class
# class ImageDataset(Dataset):
#     def __init__(self, image_paths, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert('L')
#         if self.transform:
#             image = self.transform(image)
#         return image

# # Transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((128, 128))
# ])

# # Load dataset
# image_paths = ['path/to/your/image1.png', 'path/to/your/image2.png']  # Add your image paths here
# dataset = ImageDataset(image_paths, transform=transform)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Initialize model, loss function, and optimizer
# model = UNet()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for images in dataloader:
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, images)  # Assuming ground truth masks are the same as input images for simplicity

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Save the model
# torch.save(model.state_dict(), 'unet_model.pth')

# # Inference
# def predict(image_path, model, transform):
#     model.eval()
#     image = Image.open(image_path).convert('L')
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(image)
#     mask = output.squeeze().numpy()
#     return mask

# # Example usage
# mask = predict("/Users/torbeng/Documents/code/proki/proki-2024/data/evaluate/1/part_1.png", model, transform)
# Image.fromarray((mask * 255).astype(np.uint8)).save('output_mask.png')

# import cv2
# import numpy as np

# def is_valid_position(center, angle, saugknopf_positions, contours, greifer_radius):
#     # Rotierte Saugknöpfe berechnen
#     rotated_positions = [
#         (
#             center[0] + x * np.cos(angle) - y * np.sin(angle),
#             center[1] + x * np.sin(angle) + y * np.cos(angle),
#         )
#         for x, y in saugknopf_positions
#     ]
#     # Prüfen, ob alle rotierten Saugknöpfe gültig sind
#     for pos in rotated_positions:
#         for contour in contours:
#             if cv2.pointPolygonTest(contour, pos, True) <= greifer_radius:
#                 return False
#     return True

import cv2


def find_optimal_position(image, saugknopf_positions, greifer_radius, step_size=10, angle_step=15):
    # Vorverarbeitung
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Bildmittelpunkt
    image_center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)
    best_position = None
    best_angle = None
    min_distance = float('inf')
    
    # Rasterbasierte Suche
    for y in range(0, binary_image.shape[0], step_size):
        for x in range(0, binary_image.shape[1], step_size):
            for angle in range(0, 360, angle_step):
                # if is_valid_position((x, y), np.radians(angle), saugknopf_positions, contours, greifer_radius):
                if is_valid_configuration(greifer_radius, saugknopf_positions, x, y, np.radians(angle)):
                    # Abstand zum Bildmittelpunkt berechnen
                    distance = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
                    if distance < min_distance:
                        best_position = (x, y)
                        best_angle = angle
                        min_distance = distance

    return best_position, best_angle

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
