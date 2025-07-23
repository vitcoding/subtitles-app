import logging
import subprocess
import time

logger = logging.getLogger(__name__)


def add_subtitles_to_video(
    input_video: str,
    input_srt: str,
    output_video: str,
    hardcode: bool = True,
    subtitle_style: str = "FontName=Arial,FontSize=14,PrimaryColour=&HFFFFFF&",
) -> bool:
    """
    Adds subtitles to a video file using FFmpeg.
    """

    start = time.perf_counter()
    try:
        if hardcode:
            # Burn subtitles into video (hardcode)
            cmd = [
                "ffmpeg",
                "-i",
                input_video,
                "-vf",
                f"subtitles={input_srt}:force_style='{subtitle_style}'",
                "-c:a",
                "copy",  # Copy audio without re-encoding
                output_video,
                "-y",  # Overwrite output file if exists
            ]
        else:
            # Add subtitles as separate stream (soft subtitles)
            cmd = [
                "ffmpeg",
                "-i",
                input_video,
                "-i",
                input_srt,
                "-c:v",
                "copy",  # Copy video stream without re-encoding
                "-c:a",
                "copy",  # Copy audio stream without re-encoding
                "-c:s",
                "mov_text",  # Subtitle codec for MP4
                "-metadata:s:s:0",
                "language=eng",  # Subtitle language (optional)
                output_video,
                "-y",
            ]

        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        end = time.perf_counter()
        video_subtitles_time = end - start
        logger.info(
            f"Subtitles added to a video in {video_subtitles_time} seconds"
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ An error while adding subtitles: {e}")
        return False
