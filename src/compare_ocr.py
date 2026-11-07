
import os
import time
import glob
import json

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import torch
import pandas as pd
import pytesseract

from PIL import Image
from jiwer import wer, cer
from paddleocr import PaddleOCR

print("OCR comparison script started", flush=True)
print("PyTorch version:", torch.__version__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REFERENCE_FILE = os.path.join(OUTPUT_DIR, "reference_text.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

MAX_FRAMES = 30

PADDLE_MODELS = ["PP-OCRv4", "PP-OCRv5", "PP-OCRv6"]


def clean_text(text):
    return " ".join(str(text or "").split()).strip()


def find_frames():
    paths = []

    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(FRAMES_DIR, extension)))

    return sorted(paths)


def run_tesseract(image_path):
    start = time.perf_counter()

    with Image.open(image_path) as image:
        data = pytesseract.image_to_data(
            image.convert("RGB"),
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
    average_confidence = sum(scores) / len(scores) if scores else None

    return text, average_confidence, elapsed


def make_paddle_engine(version):
    return PaddleOCR(
        lang="en",
        ocr_version=version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False
    )


def run_paddle(engine, image_path):
    start = time.perf_counter()
    results = engine.predict(image_path)

    texts = []
    scores = []

    for item in results:
        data = getattr(item, "json", None)

        if callable(data):
            data = data()

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                continue

        if isinstance(data, dict):
            data = data.get("res", data)

            texts.extend(
                str(text) for text in data.get("rec_texts", [])
            )

            for score in data.get("rec_scores", []):
                try:
                    scores.append(float(score) * 100.0)
                except (TypeError, ValueError):
                    pass

    text = clean_text(" ".join(texts))
    average_confidence = sum(scores) / len(scores) if scores else None
    elapsed = time.perf_counter() - start

    return text, average_confidence, elapsed


def load_references():
    references = {}

    if not os.path.exists(REFERENCE_FILE):
        print("\nNo reference CSV found.")
        print("WER/CER requires manually verified reference text.")
        print("Expected file: output\\reference_text.csv")
        return references

    try:
        ref_df = pd.read_csv(REFERENCE_FILE).fillna("")
    except Exception as exc:
        print("Could not read reference CSV:", exc)
        return references

    if not {"Frame", "Reference_Text"}.issubset(ref_df.columns):
        print("Reference CSV needs Frame and Reference_Text columns.")
        return references

    for _, row in ref_df.iterrows():
        references[os.path.basename(str(row["Frame"]))] = clean_text(
            row["Reference_Text"]
        )

    print("Reference texts loaded:", len(references))
    return references


def main():
    print("Project folder:", BASE_DIR)
    print("Frames folder:", FRAMES_DIR)

    frame_paths = find_frames()

    if not frame_paths:
        print("ERROR: No frames found in:", FRAMES_DIR)
        return

    if MAX_FRAMES is not None:
        frame_paths = frame_paths[:MAX_FRAMES]

    print("Frames selected:", len(frame_paths))

    references = load_references()
    records = []

    print("\n========== Testing Tesseract ==========")

    for index, path in enumerate(frame_paths, start=1):
        filename = os.path.basename(path)

        print(
            f"Tesseract: {index}/{len(frame_paths)} - {filename}",
            flush=True
        )

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
            print(f"Tesseract error on {filename}: {exc}")

            records.append({
                "Frame": filename,
                "Model": "Tesseract",
                "OCR_Text": "",
                "Confidence": None,
                "Time_sec": None,
                "Error": str(exc)
            })

    for version in PADDLE_MODELS:
        print(f"\n========== Loading {version} ==========", flush=True)

        try:
            engine = make_paddle_engine(version)
            print(f"{version} loaded successfully.", flush=True)

        except Exception as exc:
            message = (
                f"Initialization failed: {type(exc).__name__}: {exc}"
            )

            print(f"Could not load {version}: {message}", flush=True)

            for path in frame_paths:
                records.append({
                    "Frame": os.path.basename(path),
                    "Model": version,
                    "OCR_Text": "",
                    "Confidence": None,
                    "Time_sec": None,
                    "Error": message
                })

            continue

        for index, path in enumerate(frame_paths, start=1):
            filename = os.path.basename(path)

            print(
                f"{version}: {index}/{len(frame_paths)} - {filename}",
                flush=True
            )

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
                message = f"{type(exc).__name__}: {exc}"

                print(
                    f"{version} error on {filename}: {message}",
                    flush=True
                )

                records.append({
                    "Frame": filename,
                    "Model": version,
                    "OCR_Text": "",
                    "Confidence": None,
                    "Time_sec": None,
                    "Error": message
                })

        del engine

    if not records:
        print("No OCR results were produced.")
        return

    details = pd.DataFrame(records)
    details["Reference_Text"] = (
        details["Frame"].map(references).fillna("")
    )

    details["WER_percent"] = None
    details["CER_percent"] = None
    details["Word_accuracy_estimate_percent"] = None

    for index, row in details.iterrows():
        reference = row["Reference_Text"]

        if reference.strip() and not row["Error"]:
            word_error = wer(reference, row["OCR_Text"])
            char_error = cer(reference, row["OCR_Text"])

            details.at[index, "WER_percent"] = word_error * 100
            details.at[index, "CER_percent"] = char_error * 100
            details.at[index, "Word_accuracy_estimate_percent"] = (
                100 - word_error * 100
            )

    details_path = os.path.join(
        OUTPUT_DIR,
        "ocr_comparison_details.csv"
    )

    details.to_csv(details_path, index=False)

    summary = details.groupby("Model").agg(
        Frames_tested=("Frame", "nunique"),
        Successful_frames=("Error", lambda values: (values == "").sum()),
        Average_time_sec=("Time_sec", "mean"),
        Average_confidence=("Confidence", "mean"),
        Errors=("Error", lambda values: (values != "").sum())
    ).reset_index()

    accuracy_results = []

    for model in summary["Model"]:
        subset = details[
            (details["Model"] == model)
            & (details["Reference_Text"].str.strip() != "")
            & (details["Error"] == "")
        ]

        if not subset.empty:
            reference = "\n".join(
                subset["Reference_Text"].astype(str)
            )

            prediction = "\n".join(
                subset["OCR_Text"].astype(str)
            )

            word_error = wer(reference, prediction)
            char_error = cer(reference, prediction)

            accuracy_results.append({
                "Model": model,
                "Reference_frames": len(subset),
                "WER_percent": word_error * 100,
                "CER_percent": char_error * 100,
                "Word_accuracy_estimate_percent": (
                    100 - word_error * 100
                )
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
        OUTPUT_DIR,
        "ocr_comparison_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== OCR COMPARISON SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nDetailed results:", details_path)
    print("Summary results:", summary_path)

    print("\nIMPORTANT:")
    print("- Confidence is not accuracy.")
    print("- WER/CER requires verified reference text.")
    print("- All models use the same selected frames.")
    print("- Check the Error column before comparing models.")


if __name__ == "__main__":
    main()
