
import csv
import os
import re
import difflib
import string

INPUT_FILE = r"output\slide_text.csv"
OUTPUT_FILE = r"output\slide_text_clean.csv"

MIN_CONFIDENCE = 40
MIN_TEXT_LENGTH = 10
MAX_FRAME_GAP = 3
SIMILARITY_THRESHOLD = 0.82


def get_frame_number(filename):
    match = re.search(r"frame_(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def similarity(text1, text2):
    a = normalize_text(text1)
    b = normalize_text(text2)

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found:", INPUT_FILE)
        return

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("Slide_Text", "").strip()

            try:
                confidence = float(row.get("OCR_Confidence", 0))
            except ValueError:
                continue

            frame_number = get_frame_number(row.get("Frame", ""))

            if frame_number < 0:
                continue

            if confidence < MIN_CONFIDENCE or len(text) < MIN_TEXT_LENGTH:
                continue

            records.append({
                "Frame": row["Frame"],
                "frame_number": frame_number,
                "OCR_Confidence": confidence,
                "Slide_Text": text
            })

    records.sort(key=lambda r: r["frame_number"])

    groups = []

    for record in records:
        matched_group = None

        # Compare against recent groups, not just the previous frame.
        for group in reversed(groups):
            gap = record["frame_number"] - group[-1]["frame_number"]

            if gap > MAX_FRAME_GAP:
                break

            best_similarity = max(
                similarity(record["Slide_Text"], member["Slide_Text"])
                for member in group
            )

            if best_similarity >= SIMILARITY_THRESHOLD:
                matched_group = group
                break

        if matched_group is None:
            groups.append([record])
        else:
            matched_group.append(record)

    results = []

    for slide_id, group in enumerate(groups, start=1):
        best = max(group, key=lambda r: r["OCR_Confidence"])

        results.append({
            "Slide_ID": slide_id,
            "Frame": best["Frame"],
            "OCR_Confidence": round(best["OCR_Confidence"], 2),
            "Slide_Text": best["Slide_Text"]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Slide_ID", "Frame", "OCR_Confidence", "Slide_Text"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("Useful OCR records:", len(records))
    print("Slide groups:", len(groups))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
