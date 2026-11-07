
import csv
import os
import re

TRANSCRIPT_FILE = r"output\transcription.txt"
SLIDE_FILE = r"output\slide_text_timestamped.csv"
OUTPUT_FILE = r"output\combined_output.csv"

BEFORE_SLIDE_SECONDS = 5
AFTER_SLIDE_SECONDS = 20


def timestamp_to_seconds(timestamp):
    parts = [int(p) for p in timestamp.split(":")]

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    raise ValueError(f"Invalid timestamp: {timestamp}")


def main():
    if not os.path.exists(TRANSCRIPT_FILE):
        print("Transcript file not found:", TRANSCRIPT_FILE)
        return

    if not os.path.exists(SLIDE_FILE):
        print("Timestamped slide file not found:", SLIDE_FILE)
        print("Run src\\timestamps.py first.")
        return

    speech_segments = []

    with open(TRANSCRIPT_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            match = re.match(
                r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)",
                line
            )

            if match:
                speech_segments.append({
                    "start": float(match.group(1)),
                    "end": float(match.group(2)),
                    "text": match.group(3).strip()
                })

    slides = []

    with open(SLIDE_FILE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                slide_seconds = timestamp_to_seconds(row["Timestamp"])
            except (ValueError, KeyError):
                continue

            slides.append({
                "slide_id": row["Slide_ID"],
                "timestamp": row["Timestamp"],
                "time_seconds": slide_seconds,
                "frame": row["Frame"],
                "confidence": row["OCR_Confidence"],
                "text": row["Slide_Text"]
            })

    slides.sort(key=lambda s: s["time_seconds"])
    speech_segments.sort(key=lambda s: s["start"])

    results = []

    for slide in slides:
        window_start = max(
            0, slide["time_seconds"] - BEFORE_SLIDE_SECONDS
        )
        window_end = slide["time_seconds"] + AFTER_SLIDE_SECONDS

        matching_speech = [
            segment["text"]
            for segment in speech_segments
            if segment["end"] >= window_start
            and segment["start"] <= window_end
            and segment["text"]
        ]

        results.append([
            slide["slide_id"],
            slide["timestamp"],
            slide["frame"],
            slide["confidence"],
            " ".join(matching_speech),
            slide["text"]
        ])

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Slide_ID", "Timestamp", "Frame",
            "OCR_Confidence", "Speech", "Slide_Text"
        ])
        writer.writerows(results)

    print("Speech segments:", len(speech_segments))
    print("Slides:", len(slides))
    print("Combined records:", len(results))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
