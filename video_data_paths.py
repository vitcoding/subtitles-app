import os


def get_paths(input_video: str) -> tuple[str]:
    """Get file paths."""
    video_dir_path = os.path.dirname(input_video)
    output_dir_path = f"{video_dir_path}/output"
    audio_path = f"{output_dir_path}/temp_audio.wav"
    srt_path = f"{output_dir_path}/temp_subtitles.srt"
    output_video_path = f"{output_dir_path}/video_with_subs.mp4"
    return audio_path, srt_path, output_video_path
