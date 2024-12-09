import os
from pathlib import Path
import pandas as pd
import re

def find_gripper(ext, part_folder):
    return next((f for f in part_folder.iterdir() if f.is_file() and re.match(r'gripper_', f.name)), None)

def generate_csv(folder_path: str, output_csv: str):
    """
    Generate a csv file, which can be used as an input for the actual hackathon thingy.

    :param folder_path: Path to the folder containing folders part_x.
    :param output_csv: Path to save the generated CSV file.
    """
    folder_path = Path(folder_path)
    assert folder_path.exists(), f"Folder {folder_path} not found."

    rows = []

    for part_folder in folder_path.iterdir():
        print( folder_path.iterdir())
        if part_folder.is_dir() and part_folder.name.startswith("part_") or part_folder.name.isnumeric():
            # Identify the file with only a number as the name
            part_file = find_gripper("png", part_folder)

            if not part_file:
                part_file = find_gripper("svg", part_folder)
                if not part_file:
                    print(f"Part not found {part_folder}. File will be skipped.")
                    continue

            # Identify all mask files
            mask_files = [f for f in part_folder.iterdir() if f.is_file() and f.name.startswith("part_")]

            # Add rows to the output for each mask file
            for mask_file in mask_files:
                rows.append([str(part_file), str(mask_file)])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows, columns=["part", "gripper"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    generate_csv("FOLDER PATH", "generated_input_file_eva.csv")
