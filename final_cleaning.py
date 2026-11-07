import pandas as pd
import re
import os

input_file = "output/combined_output.csv"
output_file = "output/final_output.csv"

# ----------------------------------------
# Load combined data
# ----------------------------------------

df = pd.read_csv(input_file)

print("Records loaded:", len(df))

# ----------------------------------------
# Clean OCR text
# ----------------------------------------

def clean_ocr(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Replace common OCR characters
    replacements = {
        "|": " ",
        "—": " ",
        "–": " ",
        "_": " ",
        "~": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # Remove isolated single characters
    words = text.split()

    cleaned_words = []

    for word in words:

        # Keep useful short scientific words
        if len(word) == 1 and word.lower() not in ["a", "i"]:
            continue

        cleaned_words.append(word)

    text = " ".join(cleaned_words)

    return text.strip()


# ----------------------------------------
# Clean speech
# ----------------------------------------

def clean_speech(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["Slide_Text"] = df["Slide_Text"].apply(clean_ocr)

df["Speech"] = df["Speech"].apply(clean_speech)

# ----------------------------------------
# Remove very poor OCR records
# ----------------------------------------

def useful_ocr(text):

    if not text:
        return False

    words = text.split()

    # Need at least 2 words
    if len(words) < 2:
        return False

    # Count alphabetic words
    alphabetic_words = [
        w for w in words
        if re.search(r"[A-Za-z]", w)
    ]

    if len(alphabetic_words) < 2:
        return False

    return True


df = df[df["Slide_Text"].apply(useful_ocr)]

# ----------------------------------------
# Create topic using slide + speech text
# ----------------------------------------

def find_topic(row):

    text = (
        str(row["Slide_Text"]) + " " +
        str(row["Speech"])
    ).lower()

    if any(word in text for word in [
        "gene", "genome", "genomics"
    ]):
        return "Genes and Genomics"

    if any(word in text for word in [
        "dna", "watson", "crick",
        "franklin", "double helix"
    ]):
        return "DNA Structure"

    if any(word in text for word in [
        "cell migration", "migration"
    ]):
        return "Cell Migration"

    if any(word in text for word in [
        "tissue", "myosin", "membrane",
        "fold", "sculpt"
    ]):
        return "Tissue Formation"

    if any(word in text for word in [
        "origin", "origins of life",
        "prebiotic", "rna"
    ]):
        return "Origins of Life"

    if any(word in text for word in [
        "organism", "organisms"
    ]):
        return "Organisms"

    if any(word in text for word in [
        "cell", "cells"
    ]):
        return "Cells"

    return "General Biology"


df["Topic"] = df.apply(find_topic, axis=1)

# ----------------------------------------
# Keep final useful columns
# ----------------------------------------

df = df[
    [
        "Slide_ID",
        "Timestamp",
        "Frame",
        "OCR_Confidence",
        "Topic",
        "Slide_Text",
        "Speech"
    ]
]

# ----------------------------------------
# Save final CSV
# ----------------------------------------

os.makedirs("output", exist_ok=True)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Final cleaning completed!")
print("Final records:", len(df))
print("Saved to:", output_file)