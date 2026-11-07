import subprocess
import os

video_path = r"videos\biology.mp4"
audio_path = r"output\audio.wav"

os.makedirs("output", exist_ok=True)

# Your FFmpeg executable
ffmpeg_path = (
    r"C:\Users\prasa\Downloads\ffmpeg-8.1.2-essentials_build"
    r"\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
)

print("Extracting audio...")

command = [
    ffmpeg_path,
    "-y",
    "-i", video_path,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    audio_path
]

result = subprocess.run(command)

if result.returncode == 0:
    print("Audio extracted successfully!")
    print("Saved to:", audio_path)
else:
    print("Audio extraction failed.")
