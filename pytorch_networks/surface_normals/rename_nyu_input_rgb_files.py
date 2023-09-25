import os
import argparse


def rename_nyu_input_rgb_files(ARGS):
    list_input_rgb_files = os.listdir(ARGS.dir_input_rgb)

    print("renaming NYU input RGB files")
    print(f"number of input RGB files: {len(list_input_rgb_files)}")

    for file_input_rgb in list_input_rgb_files:
        # load sur nor label in RGB format
        file_new_rgb = file_input_rgb.replace("_rgb", "").split(".")[0] + "-rgb.png"
        os.rename(
            os.path.join(ARGS.dir_input_rgb, file_input_rgb),
            os.path.join(ARGS.dir_input_rgb, file_new_rgb),
        )
    print(f"renamed all the NYU input RGB files in: {ARGS.dir_input_rgb}")
    return


def main():
    parser = argparse.ArgumentParser(
        description="rename NYU input RGB files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir_input_rgb", type=str,
        required=True, help="full path to directory with NYU input RGB files")
    ARGS = parser.parse_args()
    rename_nyu_input_rgb_files(ARGS)
    return

if __name__ == "__main__":
    main()
