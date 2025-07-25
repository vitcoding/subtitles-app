import logging
import os
import time
from typing import List, Tuple

import pysrt
from faster_whisper import WhisperModel
from pydub import AudioSegment

from subtitles.subtitles_change import process_srt_file
from video_data_paths import get_paths

logger = logging.getLogger(__name__)


def transcribe_audio_with_whisper(
    audio_path: str,
    model_size: str = "small",
    language: str = "ru",
    device: str = "cpu",
) -> Tuple[List[dict], dict]:
    """
    Transcribes audio using Faster Whisper model.
    """
    try:
        # Initialize Whisper model with appropriate compute type
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )

        # Transcribe audio with word-level timestamps
        segments, info = model.transcribe(
            audio_path, language=language, word_timestamps=True
        )

        # Convert generator to list to ensure we can iterate multiple times
        return list(segments), info

    except Exception as e:
        logger.error(f"❌ Whisper transcription failed: {str(e)}")
        raise


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

        # Transcribe audio using Whisper
        segments, _ = transcribe_audio_with_whisper(
            temp_audio, model_size=model_size, language=language, device=device
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

                    # If no suitable group found, take first word
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
                    # Extend previous subtitle's end time
                    previous_sub.end = pysrt.SubRipTime(
                        seconds=first_word.start
                    )

                # Create new subtitle if text remains after moving punctuation
                if text:
                    new_sub = pysrt.SubRipItem(
                        index=sub_index,
                        start=pysrt.SubRipTime(seconds=first_word.start),
                        end=pysrt.SubRipTime(seconds=last_word.end),
                        text=text,
                    )
                    subs.append(new_sub)
                    previous_sub = new_sub
                    sub_index += 1

        end = time.perf_counter()
        logger.info(f"Subtitles generated in {end - start:.2f} seconds")

        # Save subtitles and apply post-processing
        subs.save(output_srt, encoding="utf-8")
        process_srt_file(output_srt, output_srt, max_gap=2.0)
        logger.info("Subtitles file updated")
        return True

    except Exception as e:
        logger.error(f"❌ Subtitle generation failed: {str(e)}")
        return False
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            logger.info("Temp audio file removed")
