import whisper
import os

audio_path = r"output\audio.wav"
output_file = r"output\transcription.txt"

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Transcribing audio...")

result = model.transcribe(audio_path)

with open(output_file, "w", encoding="utf-8") as f:

    for segment in result["segments"]:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        f.write(
            f"[{start:.2f} - {end:.2f}] {text}\n"
        )

print("Transcription completed!")
print("Saved to:", output_file)