import os
import shutil
import argparse
import csv

# slice_id mapping from the Colosseum O-RAN ColORAN metrics CSV `slice_id` column
SLICE_ID_MAPPING = {
    0: "embb",
    1: "mtc",
    2: "urllc",
}


def read_slice_id(csv_path):
    """Return the slice_id value from the first data row, or None if unreadable."""
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("slice_id")
                if value is None or value == "":
                    continue
                return int(float(value))
        return None
    except Exception as e:
        print(f"Failed to read slice_id from {csv_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Group dataset by slice_id (0=embb, 1=mtc, 2=urllc)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="Dataset/colosseum-oran-coloran-dataset",
        help="Target directory to process.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. If not set, runs in dry-run mode.",
    )
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Processing directory: {target_dir}")
    if not args.execute:
        print("--- DRY RUN MODE: No files will be actually moved ---")
        print("Run with --execute to perform operations.\n")

    stats = {"embb": 0, "mtc": 0, "urllc": 0, "unknown": 0}

    for root, _dirs, files in os.walk(target_dir):
        # skip files already inside slice subdirectories
        if any(f"/{sc}" in root for sc in ("embb", "mtc", "urllc")):
            continue

        for file in files:
            if not file.lower().endswith("_metrics.csv"):
                continue

            current_path = os.path.join(root, file)
            slice_id = read_slice_id(current_path)
            slice_type = SLICE_ID_MAPPING.get(slice_id, "unknown")
            stats[slice_type] += 1

            if slice_type == "unknown":
                continue

            if args.execute:
                slice_dir = os.path.join(root, slice_type)
                os.makedirs(slice_dir, exist_ok=True)
                new_path = os.path.join(slice_dir, file)
                try:
                    shutil.move(current_path, new_path)
                except Exception as e:
                    print(f"Failed to move {current_path}: {e}")

    print("--- Summary ---")
    for s_type, count in stats.items():
        print(f"Files for {s_type.ljust(7)}: {count}")

    if args.execute:
        print("\nSuccessfully moved files into slice-specific subdirectories.")


if __name__ == "__main__":
    main()
