import logging
import os
import time
from typing import List, Literal, Optional, Tuple, Union

import pysrt
from faster_whisper import WhisperModel
from openai import OpenAI
from pydub import AudioSegment

from response_mock_temp import response_mock
from subtitles.subtitles_change import process_srt_file
from video_data_paths import get_paths

logger = logging.getLogger(__name__)


# Type definitions
TranscriptionMethod = Literal["faster_whisper", "openai_api"]
WordType = Literal["word", "punctuation"]


class TranscriptionWord:
    """Class to represent a word with timing information"""

    def __init__(
        self, word: str, start: float, end: float, type: WordType = "word"
    ):
        self.word = word
        self.start = start
        self.end = end
        self.type = type


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: str = "ru",
    device: str = "cpu",
    method: TranscriptionMethod = "faster_whisper",
    openai_api_key: Optional[str] = None,
) -> Tuple[List[List[TranscriptionWord]], dict]:
    """
    Transcribes audio using either Faster Whisper or OpenAI API.
    Returns a tuple of (segments, info) where segments is a list of word lists with timing info.
    """
    if method not in ["faster_whisper", "openai_api"]:
        raise ValueError(f"Invalid transcription method: {method}")

    try:
        if method == "faster_whisper":
            return _transcribe_with_faster_whisper(
                audio_path, model_size, language, device
            )
        else:
            return _transcribe_with_openai_api(
                audio_path, language, openai_api_key
            )
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        raise


def _transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str,
    language: str,
    device: str,
) -> Tuple[List[List[TranscriptionWord]], dict]:
    """
    Internal function for Faster Whisper transcription.
    Converts the output to match the OpenAI API format for consistency.
    """
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    # Transcribe audio with word-level timestamps
    segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=True
    )

    # Convert segments to match OpenAI API format
    formatted_segments = []
    for segment in segments:
        words = []
        for word in segment.words:
            words.append(
                TranscriptionWord(
                    word=word.word, start=word.start, end=word.end, type="word"
                )
            )
        formatted_segments.append(words)

    return formatted_segments, {"language": info.language}


def _transcribe_with_openai_api(
    audio_path: str,
    language: str,
    api_key: str,
) -> Tuple[List[List[TranscriptionWord]], dict]:
    """
    Internal function for OpenAI API transcription.
    Handles both real API calls and mock responses for testing.
    """

    # Real API usage
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        logger.info(f"OpenAI API response: {response}")

    # For testing, use mock response
    # response = response_mock

    # Process words with attached punctuation
    processed_words = []
    text_chars = list(response.text)
    word_index = 0
    char_index = 0

    while char_index < len(text_chars) and word_index < len(response.words):
        char = text_chars[char_index]

        if char.isspace():
            char_index += 1
            continue

        current_word = response.words[word_index].word
        text_slice = "".join(
            text_chars[char_index : char_index + len(current_word)]
        )

        if text_slice.lower() == current_word.lower():
            # Find all punctuation following this word
            punctuation = []
            punct_index = char_index + len(current_word)

            while (
                punct_index < len(text_chars)
                and not text_chars[punct_index].isspace()
                and not text_chars[punct_index].isalnum()
            ):
                punctuation.append(text_chars[punct_index])
                punct_index += 1

            # Create word with attached punctuation but keep original timings
            full_word = current_word + "".join(punctuation)
            processed_words.append(
                TranscriptionWord(
                    word=full_word,
                    start=response.words[word_index].start,
                    end=response.words[
                        word_index
                    ].end,  # Keep original end time
                    type="word",
                )
            )

            char_index = punct_index
            word_index += 1
        else:
            char_index += 1

    # Return as a single segment to match faster_whisper format
    return [processed_words], {"language": response.language}


def generate_subtitles(
    video_path: str,
    output_srt: str,
    model_size: str = "small",
    language: str = "ru",
    device: str = "cpu",
    max_words: int = 3,
    max_chars: int = 17,
    long_word_threshold: int = 17,
    transcription_method: TranscriptionMethod = "faster_whisper",
    openai_api_key: Optional[str] = None,
) -> bool:
    """
    Generates subtitles with configurable transcription method and processing rules.
    Handles both local model and API responses uniformly.
    """

    try:
        start = time.perf_counter()

        # Extract audio track from video file
        audio = AudioSegment.from_file(video_path)
        audio_path, _, _ = get_paths(video_path)
        temp_audio = audio_path

        # Convert to Whisper-optimized audio format
        audio.set_frame_rate(16000).set_channels(1).export(
            temp_audio, format="wav", codec="pcm_s16le"
        )

        # Transcribe audio using selected method
        segments, _ = transcribe_audio(
            temp_audio,
            model_size=model_size,
            language=language,
            device=device,
            method=transcription_method,
            openai_api_key=openai_api_key,
        )

        subs = pysrt.SubRipFile()
        sub_index = 1
        previous_sub = None

        # Characters that should force a new subtitle line
        sentence_enders = {".", "!", "?"}
        move_to_previous_patterns = {
            '."',
            '!"',
            '?"',  # English-style quotes
            ".»",
            "!»",
            "?»",  # Russian-style quotes
        }

        for segment in segments:
            words = segment
            i = 0

            while i < len(words):
                # Handle long words as individual subtitles
                if len(words[i].word) > long_word_threshold:
                    word_group = [words[i]]
                    i += 1
                else:
                    word_group = []
                    sentence_ender_found = False
                    current_word_count = 0
                    current_char_count = 0

                    # Build word group respecting max_words and max_chars
                    while (
                        current_word_count < max_words
                        and i + current_word_count < len(words)
                    ):
                        candidate_word = words[i + current_word_count]
                        candidate_text = candidate_word.word

                        # Check if adding this word would exceed max_chars
                        if (
                            current_char_count + len(candidate_text)
                            > max_chars
                        ):
                            if (
                                current_word_count == 0
                            ):  # Single word exceeds max_chars
                                current_word_count = 1  # Take it anyway
                            break

                        # Check for sentence enders in current word
                        if any(
                            candidate_text.endswith(end)
                            for end in sentence_enders
                        ):
                            word_group.append(candidate_word)
                            current_word_count += 1
                            current_char_count += len(candidate_text)
                            sentence_ender_found = True
                            break

                        word_group.append(candidate_word)
                        current_word_count += 1
                        current_char_count += len(candidate_text)

                    i += current_word_count

                if not word_group:
                    continue

                # Create subtitle using exact timings from model
                first_word = word_group[0]
                last_word = word_group[-1]
                text = " ".join(word.word for word in word_group).strip()

                # Handle punctuation movement
                move_text = ""
                for pattern in move_to_previous_patterns:
                    if text.startswith(pattern):
                        move_text = pattern
                        text = text[len(pattern) :].strip()
                        if len(word_group) > 1:
                            first_word = word_group[1]
                        break

                if move_text and previous_sub:
                    previous_sub.text = f"{previous_sub.text}{move_text}"
                    previous_sub.end = pysrt.SubRipTime(
                        seconds=first_word.start
                    )

                if text:
                    # Use exact timings without duration adjustment
                    new_sub = pysrt.SubRipItem(
                        index=sub_index,
                        start=pysrt.SubRipTime(seconds=first_word.start),
                        end=pysrt.SubRipTime(seconds=last_word.end),
                        text=text,
                    )
                    subs.append(new_sub)
                    previous_sub = new_sub
                    sub_index += 1

                    # If we found a sentence ender, ensure next words start new subtitle
                    if sentence_ender_found:
                        previous_sub = None

        end = time.perf_counter()
        logger.info(f"Subtitles generated in {end - start:.2f} seconds")

        # Save and process subtitles
        subs.save(output_srt, encoding="utf-8")
        process_srt_file(output_srt, output_srt, max_gap=2.0)
        logger.info("Subtitles file updated")
        return True

    except Exception as e:
        logger.error(f"❌ Subtitle generation failed: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            logger.info("Temp audio file removed")
