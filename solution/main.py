from pathlib import Path
from argparse import ArgumentParser
from masker import create_part_mask

from rich.progress import track
import pandas as pd

from algorithm import findCenterOfGripper




def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    print(part_image_path)
    coords = findCenterOfGripper(part_image_path, gripper_image_path)
    if coords is None:
        print(f"Kein Schwerpunkt für {part_image_path}. Zeile überspringen.")
        return None, None, None


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
