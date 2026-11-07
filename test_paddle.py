
import os
import glob
import torch

from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

print("PyTorch version:", torch.__version__)

frame_paths = sorted(
    glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
)

if not frame_paths:
    print("ERROR: No JPG frames found in:", FRAMES_DIR)
    raise SystemExit(1)

print("\nAvailable frames:")
for path in frame_paths[:20]:
    print(" -", os.path.basename(path))

filename = input(
    "\nEnter a frame filename containing visible text "
    "(example: frame_0010.jpg): "
).strip()

image_path = os.path.join(FRAMES_DIR, filename)

if not os.path.isfile(image_path):
    print("ERROR: Frame not found:", image_path)
    raise SystemExit(1)

print("\nTesting frame:", image_path)
print("Loading PP-OCR model...")

try:
    os.environ["FLAGS_use_mkldnn"] = "0"

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False
    )

    print("Model loaded successfully.")
    print("Running OCR...")

    results = ocr.predict(image_path)

    found_text = False

    for item in results:
        data = getattr(item, "json", None)

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            print(item)
            continue

        data = data.get("res", data)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        for i, recognized_text in enumerate(texts):
            score = scores[i] if i < len(scores) else None

            print(f"Text: {recognized_text}")
            print(f"Confidence: {score}")
            print("-" * 40)

            found_text = True

    if found_text:
        print("\nSUCCESS: Text was detected.")
    else:
        print("\nNo text was detected in this frame.")
        print("Try another frame with clearly visible slide text.")

except Exception as exc:
    print("\nPP-OCR test failed.")
    print("Error type:", type(exc).__name__)
    print("Error details:", exc)
    print(
        "\nThe selected inference engine or its settings "
        "may not be supported by this installed version."
    )
