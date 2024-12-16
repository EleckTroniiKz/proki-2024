import os
from pathlib import Path
import pandas as pd
import re

def find_gripper(part_folder):
    """
    Finds the first gripper file in a folder, prioritizing PNG over SVG.
    :param part_folder: Path object pointing to the folder.
    :return: Path to the gripper file or None if not found.
    """
    # Look for PNG first
    gripper_file = next(
        (f for f in part_folder.iterdir() if f.is_file() and re.match(r'^gripper_\d+\.png$', f.name)),
        None
    )
    # If no PNG found, look for SVG
    if not gripper_file:
        gripper_file = next(
            (f for f in part_folder.iterdir() if f.is_file() and re.match(r'^gripper_\d+\.svg$', f.name)),
            None
        )
    return gripper_file

def generate_csv(parent_folder: str, output_csv: str):
    """
    Generate a CSV file listing gripper-part pairs for all subfolders.

    :param parent_folder: Path to the parent folder containing subfolders.
    :param output_csv: Path to save the generated CSV file.
    """
    parent_folder = Path(parent_folder)
    assert parent_folder.exists(), f"Parent folder {parent_folder} not found."

    rows = []

    # Iterate through each subfolder
    for part_folder in parent_folder.iterdir():
        if part_folder.is_dir():
            # Find all gripper files in the folder
            gripper_files = [
                find_gripper(part_folder)
                for f in part_folder.iterdir()
                if re.match(r'^gripper_\d+\.(png|svg)$', f.name)
            ]
            gripper_files = list(set([g for g in gripper_files if g]))

            if not gripper_files:
                print(f"No gripper files found in {part_folder}. Skipping folder.")
                continue

            # Find all part files in the folder
            part_files = [
                f for f in part_folder.iterdir()
                if f.is_file() and re.match(r'^part_\d+\.(png|svg)$', f.name)
            ]

            if not part_files:
                print(f"No part files found in {part_folder}. Skipping folder.")
                continue

            # Add all gripper-part combinations to the rows
            for gripper_file in gripper_files:
                for part_file in part_files:
                    rows.append([str(gripper_file), str(part_file)])

    # Create DataFrame and save to CSV
    if rows:
        df = pd.DataFrame(rows, columns=["gripper", "part"])
        df.to_csv(output_csv, index=False)
        print(f"CSV file saved to {output_csv}")
    else:
        print("No valid gripper-part pairs found. No CSV file generated.")

# Example usage
if __name__ == "__main__":
    generate_csv("YOUR_PARENT_FOLDER", "FILE_NAME_FOR_OUTPUT")
    
    """
    Make sure that your first parameter for generate_csv is a parent folder, of folders. And those subfolders contain atleast one gripper and one part.
    """
