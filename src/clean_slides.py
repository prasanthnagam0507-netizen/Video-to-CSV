import csv
import os
import re

INPUT_FILE = r"output\slide_text_timestamped.csv"
OUTPUT_FILE = r"output\slide_text_final.csv"


def clean_ocr_text(text):

    # Replace common OCR characters
    replacements = {
        "|": " ",
        "~": " ",
        "_": " ",
        "=": " ",
        "—": " ",
        "–": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces around punctuation
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    return text.strip()


def is_useful(text):

    if not text:
        return False

    # Count alphabetic characters
    letters = sum(c.isalpha() for c in text)

    # Count total characters
    total = len(text)

    if total == 0:
        return False

    letter_ratio = letters / total

    # Require reasonable amount of readable text
    if letters < 8:
        return False

    if letter_ratio < 0.35:
        return False

    return True


if not os.path.exists(INPUT_FILE):

    print("Input file not found:")
    print(INPUT_FILE)
    exit()


results = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        text = clean_ocr_text(
            row["Slide_Text"]
        )

        confidence = float(
            row["OCR_Confidence"]
        )

        if is_useful(text):

            results.append([
                row["Slide_ID"],
                row["Timestamp"],
                row["Frame"],
                round(confidence, 2),
                text
            ])


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Slide_ID",
        "Timestamp",
        "Frame",
        "OCR_Confidence",
        "Slide_Text"
    ])

    writer.writerows(results)


print()
print("OCR cleaning completed!")
print("Original records:", len(list(
    csv.DictReader(
        open(
            INPUT_FILE,
            encoding="utf-8-sig"
        )
    )
)))

print("Useful cleaned records:", len(results))
print("Saved to:", OUTPUT_FILE)
