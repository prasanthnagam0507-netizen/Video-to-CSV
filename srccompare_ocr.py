
import os
import time
import glob
import json
import pandas as pd
import pytesseract

from PIL import Image
from jiwer import wer, cer
from paddleocr import PaddleOCR

print("OCR comparison script started", flush=True)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REFERENCE_FILE = os.path.join(OUTPUT_DIR, "reference_text.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Test a limited number of frames first.
# Set to None to test all frames.
MAX_FRAMES = 30

# Model identifiers depend on your PaddleOCR version.
PADDLE_MODELS = ["PP-OCRv4", "PP-OCRv5", "PP-OCRv6"]


def clean_text(text):
    return " ".join(str(text or "").split()).strip()


def find_frames():
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []

    for ext in extensions:
        paths.extend(glob.glob(os.path.join(FRAMES_DIR, ext)))

    return sorted(paths)


def run_tesseract(image_path):
    start = time.perf_counter()

    data = pytesseract.image_to_data(
        Image.open(image_path).convert("RGB"),
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    scores = []

    for word, confidence in zip(data["text"], data["conf"]):
        word = word.strip()

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            continue

        if word and confidence >= 40:
            words.append(word)
            scores.append(confidence)

    elapsed = time.perf_counter() - start
    text = clean_text(" ".join(words))
    avg_confidence = sum(scores) / len(scores) if scores else None

    return text, avg_confidence, elapsed


def run_paddle(engine, image_path):
    start = time.perf_counter()
    result = engine.predict(image_path)

    texts = []
    scores = []

    for item in result:
        data = getattr(item, "json", None)

        if callable(data):
            data = data()

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):
            data = data.get("res", data)
            texts.extend(data.get("rec_texts", []))
            scores.extend(data.get("rec_scores", []))

    text = clean_text(" ".join(texts))
    confidence = (
        sum(float(s) for s in scores) / len(scores)
        if scores else None
    )
    elapsed = time.perf_counter() - start

    return text, confidence, elapsed


def main():
    print("Project folder:", BASE_DIR)
    print("Frames folder:", FRAMES_DIR)

    frame_paths = find_frames()

    if not frame_paths:
        print("ERROR: No frames found.")
        print("Check whether your images are stored in:", FRAMES_DIR)
        print("If your frames are inside another folder, update FRAMES_DIR.")
        return

    if MAX_FRAMES is not None:
        frame_paths = frame_paths[:MAX_FRAMES]

    print("Frames selected:", len(frame_paths))

    references = {}

    if os.path.exists(REFERENCE_FILE):
        ref_df = pd.read_csv(REFERENCE_FILE).fillna("")

        if {"Frame", "Reference_Text"}.issubset(ref_df.columns):
            references = {
                os.path.basename(str(row["Frame"])): clean_text(
                    row["Reference_Text"]
                )
                for _, row in ref_df.iterrows()
            }
            print("Reference texts loaded:", len(references))
        else:
            print("Reference CSV needs Frame and Reference_Text columns.")
    else:
        print("No reference CSV found.")
        print("WER/CER cannot be measured until verified reference text is provided.")

    records = []

    # Test Tesseract
    print("\nTesting Tesseract...")

    for index, path in enumerate(frame_paths, start=1):
        filename = os.path.basename(path)
        print(f"Tesseract: {index}/{len(frame_paths)} - {filename}")

        try:
            text, confidence, elapsed = run_tesseract(path)

            records.append({
                "Frame": filename,
                "Model": "Tesseract",
                "OCR_Text": text,
                "Confidence": confidence,
                "Time_sec": elapsed,
                "Error": ""
            })

        except Exception as exc:
            print("Tesseract error:", exc)

            records.append({
                "Frame": filename,
                "Model": "Tesseract",
                "OCR_Text": "",
                "Confidence": None,
                "Time_sec": None,
                "Error": str(exc)
            })

    # Test PaddleOCR versions
    for version in PADDLE_MODELS:
        print(f"\nLoading {version}...")

        try:
            engine = PaddleOCR(
                lang="en",
                ocr_version=version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )

        except Exception as exc:
            print(f"Could not load {version}: {exc}")
            print("Skipping this model.")
            continue

        for index, path in enumerate(frame_paths, start=1):
            filename = os.path.basename(path)
            print(f"{version}: {index}/{len(frame_paths)} - {filename}")

            try:
                text, confidence, elapsed = run_paddle(engine, path)

                records.append({
                    "Frame": filename,
                    "Model": version,
                    "OCR_Text": text,
                    "Confidence": confidence,
                    "Time_sec": elapsed,
                    "Error": ""
                })

            except Exception as exc:
                print(f"{version} error on {filename}: {exc}")

                records.append({
                    "Frame": filename,
                    "Model": version,
                    "OCR_Text": "",
                    "Confidence": None,
                    "Time_sec": None,
                    "Error": str(exc)
                })

        del engine

    if not records:
        print("No OCR results were produced.")
        return

    details = pd.DataFrame(records)
    details["Reference_Text"] = details["Frame"].map(references).fillna("")

    details["WER_percent"] = None
    details["CER_percent"] = None
    details["Word_accuracy_estimate_percent"] = None

    # Calculate errors only for frames with verified reference text
    for index, row in details.iterrows():
        reference = row["Reference_Text"]
        prediction = row["OCR_Text"]

        if reference.strip():
            word_error = wer(reference, prediction)
            char_error = cer(reference, prediction)

            details.at[index, "WER_percent"] = word_error * 100
            details.at[index, "CER_percent"] = char_error * 100
            details.at[index, "Word_accuracy_estimate_percent"] = (
                100 - word_error * 100
            )

    details_path = os.path.join(
        OUTPUT_DIR, "ocr_comparison_details.csv"
    )
    details.to_csv(details_path, index=False)

    # Summarize time and confidence by model
    summary = details.groupby("Model").agg(
        Frames_tested=("Frame", "nunique"),
        Average_time_sec=("Time_sec", "mean"),
        Average_confidence=("Confidence", "mean"),
        Errors=("Error", lambda values: (values != "").sum())
    ).reset_index()

    # Aggregate WER/CER over frames that have reference text
    accuracy_results = []

    for model in summary["Model"]:
        subset = details[
            (details["Model"] == model)
            & (details["Reference_Text"].str.strip() != "")
        ]

        if not subset.empty:
            reference = "\n".join(subset["Reference_Text"].astype(str))
            prediction = "\n".join(subset["OCR_Text"].astype(str))

            word_error = wer(reference, prediction)
            char_error = cer(reference, prediction)

            accuracy_results.append({
                "Model": model,
                "Reference_frames": len(subset),
                "WER_percent": word_error * 100,
                "CER_percent": char_error * 100,
                "Word_accuracy_estimate_percent": 100 - word_error * 100
            })

    if accuracy_results:
        summary = summary.merge(
            pd.DataFrame(accuracy_results),
            on="Model",
            how="left"
        )
    else:
        summary["Reference_frames"] = 0
        summary["WER_percent"] = None
        summary["CER_percent"] = None
        summary["Word_accuracy_estimate_percent"] = None

    summary_path = os.path.join(
        OUTPUT_DIR, "ocr_comparison_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n========== OCR COMPARISON ==========")
    print(summary.to_string(index=False))
    print("\nDetailed results:", details_path)
    print("Summary results:", summary_path)

    print("\nIMPORTANT:")
    print("Confidence is not accuracy.")
    print("Actual WER/CER requires verified reference text.")
    print("Compare models only on the same frames and reference text.")


if __name__ == "__main__":
    main()
