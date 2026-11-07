
import os
import re
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------
# Configuration
# ----------------------------------------

INPUT_FILE = r"output\combined_output.csv"
OUTPUT_FILE = r"output\information_output.csv"
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

os.makedirs("output", exist_ok=True)

# ----------------------------------------
# Load input data
# ----------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run src\\combine_data.py first."
    )

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig").fillna("")

required_columns = [
    "Slide_ID",
    "Timestamp",
    "Frame",
    "OCR_Confidence",
    "Speech",
    "Slide_Text"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Records loaded:", len(df))


# ----------------------------------------
# Clean text
# ----------------------------------------

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return re.sub(r"\s+", " ", text)


df["Speech"] = df["Speech"].apply(clean_text)
df["Slide_Text"] = df["Slide_Text"].apply(clean_text)


def combined_text(row):
    return (
        clean_text(row["Speech"]) + " " +
        clean_text(row["Slide_Text"])
    ).strip()


# ----------------------------------------
# Load DistilBART summarization model
# ----------------------------------------

print("\nLoading DistilBART model...")
print("Model:", MODEL_NAME)

device = torch.device("cpu")

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("DistilBART loaded successfully!")

except Exception as error:
    raise RuntimeError(
        "Could not load DistilBART. Check your internet connection, "
        "model download, installed packages, and available disk space."
    ) from error


# ----------------------------------------
# Content type classification
# ----------------------------------------

COPYRIGHT_PATTERNS = [
    r"\bcreative commons\b",
    r"\ball rights reserved\b",
    r"\bcopyright\b",
    r"\blicensed under\b",
    r"\bclosing credits\b",
    r"\bnot covered under\b",
    r"\bcc\s*by\b",
    r"\(\s*cc\s*\)\s*by\b",
    r"\bpermission is granted\b",
    r"\bocw\.mit\.edu\b"
]

ADMIN_PATTERNS = [
    r"\bcourse organization\b",
    r"\bcourse overview\b",
    r"\bcourse schedule\b",
    r"\bgrading\b",
    r"\bgrade policy\b",
    r"\bsyllabus\b",
    r"\boffice hours\b",
    r"\brunning hours\b",
    r"\blecture schedule\b",
    r"\bassignments?\b",
    r"\bexaminations?\b",
    r"\bexam dates?\b",
    r"\battendance\b",
    r"\bcontact information\b",
    r"\bintroductions?\b",
    r"\bwelcome to\b",
    r"\bteaching staff\b",
    r"\binstructors?\b",
    r"\bannouncement\b",
    r"\bfilming will take place\b"
]


def contains_pattern(text, patterns):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def classify_content_type(row):
    speech = clean_text(row["Speech"])
    slide = clean_text(row["Slide_Text"])
    combined = (speech + " " + slide).strip()

    if contains_pattern(slide, COPYRIGHT_PATTERNS):
        return "Copyright/Attribution"

    if not combined or len(re.sub(r"\W", "", combined)) < 10:
        return "Insufficient Content"

    if contains_pattern(combined, ADMIN_PATTERNS):
        return "Course Administration"

    if len(combined.split()) < 5:
        return "Insufficient Content"

    return "Lecture Content"


df["Content_Type"] = df.apply(classify_content_type, axis=1)


# ----------------------------------------
# Topic classification
# ----------------------------------------

TOPIC_KEYWORDS = {
    "DNA Structure": [
        "dna structure", "double helix", "watson",
        "crick", "rosalind franklin", "dna replication"
    ],
    "Genes and Genomics": [
        "genome", "genomics", "genes", "gene sequencing",
        "human genome", "genetic diversity"
    ],
    "Cell Biology": [
        "cell", "cellular", "cell division",
        "cell cycle", "membrane", "myosin"
    ],
    "RNA and Early Life": [
        "rna world", "ribonucleic", "prebiotic",
        "nucleic acid", "rna"
    ],
    "Evolution": [
        "evolution", "evolved", "evolutionary",
        "homo sapiens", "human evolution"
    ],
    "Organisms and Microbiology": [
        "bacteria", "prokaryotes", "eukaryotes",
        "organisms", "cyanobacteria"
    ],
    "Microscopy and Imaging": [
        "microscopy", "fluorescent proteins",
        "imaging", "visualization", "microscope"
    ],
    "Systems and Synthetic Biology": [
        "systems biology", "synthetic biology",
        "signaling network", "protein network"
    ],
    "Proteins": [
        "proteins", "protein synthesis",
        "protein engineering", "amino acids"
    ]
}


def count_keyword_matches(text, keywords):
    score = 0

    for keyword in keywords:
        pattern = (
            r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        )
        score += len(re.findall(pattern, text.lower()))

    return score


def classify_topic(row):
    content_type = row["Content_Type"]

    if content_type in {
        "Copyright/Attribution",
        "Course Administration",
        "Insufficient Content"
    }:
        return content_type

    speech = clean_text(row["Speech"]).lower()
    slide = clean_text(row["Slide_Text"]).lower()

    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = (
            2 * count_keyword_matches(speech, keywords)
            + count_keyword_matches(slide, keywords)
        )

        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return "General Biology"

    return max(topic_scores, key=topic_scores.get)


df["Topic"] = df.apply(classify_topic, axis=1)


# ----------------------------------------
# Keyword extraction
# ----------------------------------------

KEYWORD_LIST = [
    "biology", "molecules", "cells", "cellular", "organisms",
    "genes", "genome", "genomics", "DNA", "RNA", "proteins",
    "evolution", "human genome", "human", "bacteria",
    "prokaryotes", "eukaryotes", "prebiotic", "RNA world",
    "Watson", "Crick", "Rosalind Franklin", "DNA structure",
    "cell migration", "tissue", "myosin", "membrane",
    "microscopy", "fluorescent proteins", "systems biology",
    "synthetic biology", "cell division", "cell cycle",
    "protein synthesis", "nucleic acids", "DNA replication",
    "course organization", "office hours", "syllabus",
    "assignments", "attendance", "grading",
    "creative commons", "copyright", "filming",
    "announcement"
]


def extract_keywords(row):
    if row["Content_Type"] == "Insufficient Content":
        return ""

    text = combined_text(row).lower()
    found = []

    for keyword in KEYWORD_LIST:
        pattern = (
            r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        )

        if re.search(pattern, text):
            found.append(keyword)

    return ", ".join(found)


df["Keywords"] = df.apply(extract_keywords, axis=1)


# ----------------------------------------
# DistilBART abstractive summarization
# ----------------------------------------

def create_summary(row):
    content_type = clean_text(row["Content_Type"])
    speech = clean_text(row["Speech"])
    slide = clean_text(row["Slide_Text"])

    if content_type == "Copyright/Attribution":
        return "Copyright or licensing notice."

    if content_type == "Insufficient Content":
        return "Insufficient text for summary."

    # Speech is generally more informative than slide OCR.
    if speech and slide:
        source_text = speech + " " + slide
    else:
        source_text = speech or slide

    if not source_text:
        return "Insufficient text for summary."

    # Avoid model generation for very short text.
    if len(source_text.split()) < 8:
        return source_text[:500]

    try:
        inputs = tokenizer(
            source_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        ).strip()

        if not summary:
            summary = speech or slide

        if len(summary) > 500:
            summary = summary[:497].rsplit(" ", 1)[0] + "..."

        return summary

    except Exception as error:
        print(
            f"Summarization failed for {row['Slide_ID']}: {error}"
        )

        fallback = speech or slide

        if len(fallback) > 500:
            return fallback[:497].rsplit(" ", 1)[0] + "..."

        return fallback or "Summary unavailable."


start_time = time.time()

summaries = []

total_records = len(df)

for index, (_, row) in enumerate(df.iterrows(), start=1):
    summary = create_summary(row)
    summaries.append(summary)

    if index % 10 == 0 or index == total_records:
        print(f"Summarized {index}/{total_records} records")

df["Summary"] = summaries

summarization_time = time.time() - start_time


# ----------------------------------------
# Quality indicators
# ----------------------------------------

def suspicious_ocr(text):
    text = clean_text(text)

    if len(text) < 15:
        return False

    words = re.findall(r"[A-Za-z]+", text)

    if len(words) < 4:
        return False

    one_letter_ratio = (
        sum(len(word) == 1 for word in words) / len(words)
    )

    symbol_noise = len(re.findall(r"[^\w\s]{4,}", text))

    mixed_tokens = re.findall(
        r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b", text
    )

    mixed_ratio = len(mixed_tokens) / max(len(words), 1)

    return (
        one_letter_ratio > 0.40
        or symbol_noise > 1
        or mixed_ratio > 0.25
    )


def assess_text_quality(row):
    speech = clean_text(row["Speech"])
    slide = clean_text(row["Slide_Text"])

    if not speech and not slide:
        return "No usable text"

    if slide:
        if suspicious_ocr(slide):
            return "Review OCR text"

        try:
            confidence = float(row["OCR_Confidence"])
        except (TypeError, ValueError):
            confidence = None

        if confidence is None:
            return "Review OCR confidence"

        if confidence < 50:
            return "Review OCR confidence"

    if len(speech.split()) < 6 and len(slide.split()) < 5:
        return "Review recommended"

    if len(slide) < 20 and speech:
        return "Speech available; review slide OCR"

    if not clean_text(row["Keywords"]):
        return "Review keywords"

    return "Text extracted"


df["Quality_Status"] = df.apply(
    assess_text_quality, axis=1
)


# ----------------------------------------
# Reorder and save output
# ----------------------------------------

output_columns = [
    "Slide_ID",
    "Timestamp",
    "Frame",
    "OCR_Confidence",
    "Content_Type",
    "Topic",
    "Keywords",
    "Slide_Text",
    "Speech",
    "Summary",
    "Quality_Status"
]

df = df[output_columns]

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ----------------------------------------
# Final report
# ----------------------------------------

print("\nInformation extraction completed!")
print("Records processed:", len(df))

print("\nContent type counts:")
print(df["Content_Type"].value_counts().to_string())

print("\nTopic counts:")
print(df["Topic"].value_counts().to_string())

print("\nQuality status counts:")
print(df["Quality_Status"].value_counts().to_string())

print("\nSummarization time:", round(summarization_time, 2), "seconds")

if len(df) > 0:
    print(
        "Average summary time per record:",
        round(summarization_time / len(df), 2),
        "seconds"
    )

print("Saved to:", OUTPUT_FILE)
