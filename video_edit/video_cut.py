import logging
import subprocess

logger = logging.getLogger(__name__)


def trim_video(input_file, output_file, start_time, end_time):
    """
    Trims a video file to the specified time range using FFmpeg.
    """
    try:
        # Calculate duration
        duration = float(end_time) - float(start_time)

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            input_file,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            output_file,
        ]

        # Run command
        subprocess.run(cmd, check=True, capture_output=True)

        logger.info(f"Video successfully trimmed and saved as {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode('utf-8')}")
    except Exception as e:
        logger.error(f"An error while trimming video: {str(e)}")


# Example usage
# trim_video('input.mp4', 'output.mp4', '10', '20')  # Using seconds
