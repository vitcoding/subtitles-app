import re
from typing import List, Tuple


def process_srt_file(
    input_file: str, output_file: str, max_gap: float = 2.0
) -> None:
    """
    Process an SRT subtitle file to:
    1. Make subtitle display continuous (no gaps > max_gap seconds)
    2. Convert all text to uppercase
    3. Ensure subtitles don't start exactly at 0 seconds
    4. Extend previous subtitle when gap is small (< max_gap)
    5. Remove leading punctuation marks like » and «
    """

    def time_to_ms(time_str: str) -> int:
        """Convert SRT time format (HH:MM:SS,mmm) to milliseconds."""
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def ms_to_time(ms: int) -> str:
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def clean_text(text: str) -> str:
        """Remove leading punctuation marks like » and « from text."""
        # Remove leading punctuation marks and whitespace
        cleaned = re.sub(r"^[»\s]+", "", text)
        return cleaned

    def parse_srt(content: str) -> List[Tuple[int, int, str]]:
        """Parse SRT content into list of (start_ms, end_ms, text) tuples."""
        blocks = re.split(r"\n\s*\n", content.strip())
        subtitles = []

        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                idx = lines[0].strip()
                time_range = lines[1].strip()
                text = " ".join(line.strip() for line in lines[2:])

                start_str, end_str = time_range.split(" --> ")
                start_ms = time_to_ms(start_str)
                end_ms = time_to_ms(end_str)

                subtitles.append((start_ms, end_ms, text))

        return subtitles

    def adjust_subtitles(
        subtitles: List[Tuple[int, int, str]], max_gap_ms: int
    ) -> List[Tuple[int, int, str]]:
        """
        Adjust subtitle timings with the following rules:
        1. Never start exactly at 0 seconds (use 20ms instead)
        2. For gaps < max_gap: extend previous subtitle to next start time
        3. For gaps >= max_gap: keep original timing
        """

        if not subtitles:
            return []

        adjusted = []

        # Process first subtitle
        first_start, first_end, first_text = subtitles[0]
        # Add 20ms if starting exactly at 0
        if first_start == 0:
            first_start = 20
            # Ensure we don't create invalid timing (end >= start)
            first_end = max(first_end, first_start)
        adjusted.append((first_start, first_end, first_text))

        for i in range(1, len(subtitles)):
            prev_start, prev_end, prev_text = adjusted[-1]
            curr_start, curr_end, curr_text = subtitles[i]

            gap = curr_start - prev_end

            if 0 < gap < max_gap_ms:
                # Extend previous subtitle to current start time
                adjusted[-1] = (prev_start, curr_start, prev_text)
                # Add current subtitle with original end time
                adjusted.append((curr_start, curr_end, curr_text))
            else:
                # Keep original timing for large gaps or overlaps
                adjusted.append((curr_start, curr_end, curr_text))

        return adjusted

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse and process subtitles
    subtitles = parse_srt(content)
    max_gap_ms = int(max_gap * 1000)  # Convert seconds to milliseconds
    adjusted_subtitles = adjust_subtitles(subtitles, max_gap_ms)

    # Generate output content
    output_lines = []
    for i, (start_ms, end_ms, text) in enumerate(adjusted_subtitles, 1):
        output_lines.append(f"{i}")
        output_lines.append(f"{ms_to_time(start_ms)} --> {ms_to_time(end_ms)}")
        # Apply text cleaning to remove leading punctuation marks
        cleaned_text = clean_text(text)
        output_lines.append(f"{cleaned_text.upper()}\n")

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
