import os
from pathlib import Path
import pandas as pd


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
        if part_folder.is_dir() and part_folder.name.startswith("part_"):
            # Identify the file with only a number as the name
            part_file = next((f for f in part_folder.iterdir() if f.is_file() and f.stem.isdigit()), None)

            if not part_file:
                print(f"Part not found {part_folder}. File will be skipped.")
                continue

            # Identify all mask files
            mask_files = [f for f in part_folder.iterdir() if f.is_file() and f.name.startswith("mask_")]

            # Add rows to the output for each mask file
            for mask_file in mask_files:
                rows.append([str(part_file), str(mask_file)])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows, columns=["part", "gripper"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    generate_csv("folderPathForAllParts", "generated_input_file.csv")
