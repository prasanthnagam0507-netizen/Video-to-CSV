
import os
import csv
import cv2

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Import torch before PaddleOCR to avoid the DLL issue
import torch
from paddleocr import PaddleOCR

frames_folder = "frames"
output_file = "output/slide_text.csv"

os.makedirs("output", exist_ok=True)

# PP-OCRv6 configuration
ocr = PaddleOCR(
    lang="en",
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
    enable_mkldnn=False
)

files = sorted([
    f for f in os.listdir(frames_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

results = []

print("OCR Engine: PP-OCRv6")
print("Total frames:", len(files))


def get_ocr_results(prediction):
    """Extract recognized text and confidence scores."""

    texts = []
    scores = []

    for item in prediction:
        if hasattr(item, "json"):
            data = item.json
        else:
            data = item

        # PaddleOCR 3.x may return a JSON property
        if callable(data):
            data = data()

        if isinstance(data, str):
            import json
            data = json.loads(data)

        if not isinstance(data, dict):
            continue

        # Handle the PaddleOCR 3.x result structure
        if "res" in data:
            data = data["res"]

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):
            text = str(text).strip()

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            if text and score >= 0.40:
                texts.append(text)
                scores.append(score * 100)

    return texts, scores


for i, filename in enumerate(files):

    image_path = os.path.join(frames_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print("Unable to read:", filename)
        continue

    # Resize
    image = cv2.resize(
        image,
        (1280, 720),
        interpolation=cv2.INTER_AREA
    )

    # Detect blank / solid-color frames
    gray_full = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if gray_full.std() < 20:
        print("Blank frame skipped:", filename)
        continue

    # Crop slide area
    height, width = image.shape[:2]

    crop = image[
        int(height * 0.03):int(height * 0.97),
        int(width * 0.08):int(width * 0.92)
    ]

    # PP-OCRv6 performs its own text detection and recognition.
    # Pass the color crop instead of applying Tesseract-specific
    # grayscale equalization and thresholding.
    try:
        prediction = ocr.predict(crop)
        words, confidences = get_ocr_results(prediction)

    except Exception as e:
        print("OCR error:", filename, "-", str(e))
        continue

    text = " ".join(words)
    word_count = sum(len(line.split()) for line in words)

    # Keep useful OCR
    if len(text) >= 20 and word_count >= 3:

        average_confidence = (
            sum(confidences) / len(confidences)
        ) if confidences else 0

        if average_confidence >= 40:
            results.append([
                filename,
                round(average_confidence, 2),
                text
            ])

    if (i + 1) % 10 == 0:
        print(
            f"Processed {i + 1}/{len(files)} frames"
        )

# Save CSV
with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Frame",
        "OCR_Confidence",
        "Slide_Text"
    ])

    writer.writerows(results)

print()
print("OCR completed!")
print("OCR engine: PP-OCRv6")
print("Frames containing useful text:", len(results))
print("Saved to:", output_file)
