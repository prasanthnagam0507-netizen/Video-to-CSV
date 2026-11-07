
import csv
import os
import re

INPUT_FILE = r"output\slide_text_clean.csv"
OUTPUT_FILE = r"output\slide_text_timestamped.csv"

FRAME_INTERVAL_SECONDS = 5


def get_frame_number(filename):
    match = re.search(r"frame_(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def format_timestamp(total_seconds):
    total_seconds = max(0, int(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found:", INPUT_FILE)
        return

    results = []

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frame = row["Frame"]
            frame_number = get_frame_number(frame)

            if frame_number < 0:
                continue

            timestamp = format_timestamp(
                frame_number * FRAME_INTERVAL_SECONDS
            )

            results.append([
                row["Slide_ID"],
                timestamp,
                frame,
                row["OCR_Confidence"],
                row["Slide_Text"]
            ])

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Slide_ID", "Timestamp", "Frame",
            "OCR_Confidence", "Slide_Text"
        ])
        writer.writerows(results)

    print("Timestamp mapping completed!")
    print("Slides processed:", len(results))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
