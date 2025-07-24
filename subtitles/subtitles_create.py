import logging
import os
import time

import pysrt
from faster_whisper import WhisperModel
from pydub import AudioSegment

from subtitles.subtitles_change import process_srt_file
from video_data_paths import get_paths

logger = logging.getLogger(__name__)


def generate_subtitles(
    video_path: str,
    output_srt: str,
    model_size: str = "small",
    language: str = "ru",
    device: str = "cpu",
    # device: str = "cuda",
    max_words: int = 2,  # Maximum words per subtitle (default 2)
    long_word_threshold: int = 12,  # Treat as long word if character count exceeds this
) -> bool:
    """
    Generates subtitles with configurable word limits and long word handling.
    """
    try:
        start = time.perf_counter()
        # Extract audio track from video file
        audio = AudioSegment.from_file(video_path)
        audio_path, _, _ = get_paths(video_path)
        temp_audio = audio_path

        # Convert to Whisper-optimized audio format (16kHz mono WAV)
        audio.set_frame_rate(16000).set_channels(1).export(
            temp_audio, format="wav", codec="pcm_s16le"
        )

        # Initialize Whisper model with appropriate compute type
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )

        # Transcribe audio with word-level timestamps
        segments, _ = model.transcribe(
            temp_audio, language=language, word_timestamps=True
        )

        subs = pysrt.SubRipFile()
        sub_index = 1

        # Characters that should force a new subtitle line
        sentence_enders = {".", "!", "?"}
        # sentence_enders = {".", ",", ":", ";", "!", "?"}

        for segment in segments:
            words = list(segment.words)
            i = 0

            while i < len(words):
                # Handle long words as individual subtitles
                if len(words[i].word) > long_word_threshold:
                    word_group = [words[i]]
                    i += 1
                else:
                    # Group words while respecting max_words limit and punctuation rules
                    word_group = []
                    while len(word_group) < max_words and i < len(words):
                        current_word = words[i].word

                        # If previous word ended with sentence-ending punctuation, break the group
                        if word_group and any(
                            word_group[-1].word.endswith(end)
                            for end in sentence_enders
                        ):
                            break

                        if len(current_word) > long_word_threshold:
                            break

                        word_group.append(words[i])
                        i += 1

                if not word_group:
                    continue

                # Get timestamps from first and last word in group
                first_word = word_group[0]
                last_word = word_group[-1]

                # Combine words into subtitle text
                text = " ".join(word.word for word in word_group).strip()

                # Add subtitle entry
                subs.append(
                    pysrt.SubRipItem(
                        index=sub_index,
                        start=pysrt.SubRipTime(seconds=first_word.start),
                        end=pysrt.SubRipTime(seconds=last_word.end),
                        text=text,
                    )
                )
                sub_index += 1

        end = time.perf_counter()
        logger.info(f"Subtitles generated in {end - start:.2f} seconds")

        # Save subtitles and apply post-processing
        subs.save(output_srt, encoding="utf-8")
        process_srt_file(output_srt, output_srt, max_gap=2.0)
        logger.info(f"Subtitles file updated")
        return True

    except Exception as e:
        logger.error(f"❌ Subtitle generation failed: {str(e)}")
        return False
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            logger.info(f"Temp audio file removed")
