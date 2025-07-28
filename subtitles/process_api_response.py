from openai.types.audio.transcription_verbose import TranscriptionWord


def add_punctuation_to_words(
    text: list[TranscriptionWord], words: str
) -> list[dict[str, str | float | int]]:
    """
    Updates words with punctuation marks from the original text.
    """

    words_dict = [dict(w) for w in words]

    new_words = []
    text_index = 0

    for word_data in words_dict:
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        # Find the word in text starting from current position
        word_start = text.find(word, text_index)
        if word_start == -1:
            # Word not found, add as is
            new_words.append(word_data.copy())
            continue

        # Find where the word ends in text
        word_end = word_start + len(word)

        # Look for punctuation after the word
        punctuation = ""
        if word_end < len(text):
            # Check for common punctuation marks
            while (
                word_end < len(text)
                and text[word_end] in '.,!?;:()[]{}""«»' "—–-"
            ):
                punctuation += text[word_end]
                word_end += 1

        # Create updated word with punctuation
        updated_word = {"word": word + punctuation, "start": start, "end": end}

        new_words.append(updated_word)
        text_index = word_end

    return new_words
