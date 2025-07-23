# Video Subtitle Processor

This project provides a complete pipeline for adding subtitles to videos, including audio transcription and subtitle embedding.

## Features

- Generates subtitles from video audio using speech recognition
- Embeds subtitles into the video file
- Supports both hardcoded (burned-in) and soft subtitles
- Customizable subtitle styles
- Supports multiple languages

## Requirements

- Python 3.x
- FFmpeg (must be installed and added to your system PATH)

## Installation

1. Install FFmpeg:
   - On Windows: Download from [FFmpeg official site](https://ffmpeg.org/) and add to PATH
   - On macOS: `brew install ffmpeg`
   - On Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian)

2. Clone this repository and install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your input video file in the `video_data` folder
2. Run the processing script:
   ```bash
   python main.py
   ```

## Notes

- For best results with speech recognition, use the "large" model size (requires more resources)
- The processing time depends on video length and model size
- Output video quality is preserved from the input file