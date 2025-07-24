import logging
import os

from subtitles.style_options import get_subtitle_style
from subtitles.subtitles_create import generate_subtitles
from subtitles.subtitles_to_video import add_subtitles_to_video
from video_data_paths import get_paths

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def process_video_with_subtitles(
    input_video: str,
    output_video: str | None = None,
    language: str = "ru",
    hardcode: bool = True,
    font: str = "app_subtitles/fonts/IBMPlexSans-SemiBold.ttf",
    fontsize: int = 18,
    font_colour: str = "&HFFFFFF",  # white
    marginv: int = 10,  # Vertical margin (pixels)
    model_size: str = "large",  # tiny, base, small, medium, large (large-v2, large-v3)
    keep_srt: bool = True,
    use_genererated_srt: bool = False,
):
    """
    Complete pipeline: transcribe audio and embed subtitles into video.
    """

    _, srt_path, output_video_path = get_paths(input_video)

    if output_video is None:
        output_video = output_video_path

    temp_srt = srt_path
    folder_path = os.path.dirname(temp_srt)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    flag = True
    # 1. Generate subtitles
    if use_genererated_srt:
        if not os.path.exists(temp_srt):
            flag = generate_subtitles(
                input_video, temp_srt, model_size, language
            )
    else:
        flag = generate_subtitles(input_video, temp_srt, model_size, language)

    if flag:
        subtitle_style = get_subtitle_style(
            font=font,
            fontsize=fontsize,
            primary_colour=font_colour,
            marginv=marginv,
        )

        # 2. Add subtitles to video
        success = add_subtitles_to_video(
            input_video=input_video,
            input_srt=temp_srt,
            output_video=output_video,
            hardcode=hardcode,
            subtitle_style=subtitle_style,
        )

        # 3. Cleanup
        if success:
            logger.info(f"✅ Success: Subtitled video saved to {output_video}")
        if not keep_srt and os.path.exists(temp_srt):
            os.remove(temp_srt)
            logger.info(f"Temp subtitles file removed")


if __name__ == "__main__":
    process_video_with_subtitles(
        input_video="video_data/video.mp4",
        font="app_subtitles/fonts/IBMPlexSans-SemiBold.ttf",
        fontsize=18,
        font_colour="&HFFFFFF",
        # marginv=145,
        marginv=160,
        # model_size="small",
        # keep_srt=False,
        # use_genererated_srt=True,
    )
