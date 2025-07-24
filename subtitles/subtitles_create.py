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
    max_words: int = 3,  # Default maximum words per subtitle
    max_chars: int = 20,  # Maximum characters per subtitle line
    long_word_threshold: int = 20,  # Treat as long word if character count exceeds this
) -> bool:
    """
    Generates subtitles with configurable word limits and long word handling.
    Handles punctuation sequences by moving them to previous subtitle when needed.
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

        # Initialize Whisper model
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
        previous_sub = None  # Store previous subtitle for punctuation handling

        # Characters that should force a new subtitle line
        sentence_enders = {".", "!", "?"}
        # Punctuation sequences that should be moved to previous subtitle
        move_to_previous_patterns = {
            '."',
            '!"',
            '?"',  # English-style quotes
            ".»",
            "!»",
            "?»",  # Russian-style quotes
        }

        for segment in segments:
            words = list(segment.words)
            i = 0

            while i < len(words):
                # Handle long words as individual subtitles
                if len(words[i].word) > long_word_threshold:
                    word_group = [words[i]]
                    i += 1
                else:
                    # Start with maximum allowed words
                    current_max_words = max_words
                    word_group = []

                    # Reduce word count until the line fits character limit
                    while current_max_words > 0:
                        candidate_group = words[i : i + current_max_words]

                        # Check punctuation break conditions
                        if word_group and any(
                            word_group[-1].word.endswith(end)
                            for end in sentence_enders
                        ):
                            break

                        # Check if candidate group exceeds character limit
                        candidate_text = " ".join(
                            w.word for w in candidate_group
                        ).strip()
                        if len(candidate_text) <= max_chars:
                            word_group = candidate_group
                            i += current_max_words
                            break

                        current_max_words -= 1

                    # If no suitable group found (even single word exceeds limit), take first word
                    if not word_group and i < len(words):
                        word_group = [words[i]]
                        i += 1

                if not word_group:
                    continue

                # Get timestamps from first and last word in group
                first_word = word_group[0]
                last_word = word_group[-1]

                # Combine words into subtitle text
                text = " ".join(word.word for word in word_group).strip()

                # Check if current subtitle starts with a punctuation sequence that should be moved
                move_text = ""
                for pattern in move_to_previous_patterns:
                    if text.startswith(pattern):
                        move_text = pattern
                        text = text[len(pattern) :].strip()
                        # Adjust first_word since we're moving part of the text
                        if len(word_group) > 1:
                            first_word = word_group[1]
                        break

                # If we found punctuation to move and there's a previous subtitle
                if move_text and previous_sub:
                    # Append the punctuation to previous subtitle
                    previous_sub.text = f"{previous_sub.text}{move_text}"
                    # Extend previous subtitle's end time to include this punctuation
                    previous_sub.end = pysrt.SubRipTime(
                        seconds=first_word.start
                    )

                # Create new subtitle (skip if text became empty after moving punctuation)
                if text:
                    new_sub = pysrt.SubRipItem(
                        index=sub_index,
                        start=pysrt.SubRipTime(seconds=first_word.start),
                        end=pysrt.SubRipTime(seconds=last_word.end),
                        text=text,
                    )
                    subs.append(new_sub)
                    previous_sub = (
                        new_sub  # Store reference for next iteration
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
