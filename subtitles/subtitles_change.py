import re
from typing import List, Tuple


def process_srt_file(
    input_file: str, output_file: str, max_gap: float = 2.0
) -> None:
    """
    Process an SRT subtitle file to:
    1. Make subtitle display continuous (no gaps > max_gap seconds).
    2. Convert all text to uppercase.
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

    def adjust_gaps(
        subtitles: List[Tuple[int, int, str]], max_gap_ms: int
    ) -> List[Tuple[int, int, str]]:
        """Adjust subtitle timings to eliminate gaps larger than max_gap_ms."""
        if not subtitles:
            return []

        adjusted = [subtitles[0]]

        for i in range(1, len(subtitles)):
            prev_end = adjusted[-1][1]
            curr_start = subtitles[i][0]
            gap = curr_start - prev_end

            if 0 < gap <= max_gap_ms:
                # Adjust to midpoint between subtitles
                midpoint = prev_end + gap // 2
                adjusted[-1] = (adjusted[-1][0], midpoint, adjusted[-1][2])
                adjusted.append((midpoint, subtitles[i][1], subtitles[i][2]))
            else:
                adjusted.append(subtitles[i])

        return adjusted

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse and process subtitles
    subtitles = parse_srt(content)
    max_gap_ms = int(max_gap * 1000)  # Convert seconds to milliseconds
    adjusted_subtitles = adjust_gaps(subtitles, max_gap_ms)

    # Generate output content
    output_lines = []
    for i, (start_ms, end_ms, text) in enumerate(adjusted_subtitles, 1):
        output_lines.append(f"{i}")
        output_lines.append(f"{ms_to_time(start_ms)} --> {ms_to_time(end_ms)}")
        output_lines.append(f"{text.upper()}\n")

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
