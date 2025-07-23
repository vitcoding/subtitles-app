def get_subtitle_style(
    # font: str = "Arial",
    font: str = "app_subtitles/fonts/IBMPlexSans-SemiBold.ttf",
    fontsize: int = 18,
    primary_colour: str = "&HFFFFFF",  # Main text color (white)
    secondary_colour: str = "&HFFFF00",  # Highlight color (yellow)
    outline_colour: str = "&H000000",  # Outline color (black)
    back_colour: str = "&H000000",  # Background color (black)
    bold: int = 1,  # Bold style (0=no, 1=yes)
    italic: int = 0,  # Italic style (0=no, 1=yes)
    underline: int = 0,  # Underline (0=no, 1=yes)
    strikeout: int = 0,  # Strikeout (0=no, 1=yes)
    scalex: int = 0.96,  # Horizontal scaling (%)
    scaley: int = 1,  # Vertical scaling (%)
    spacing: int = -0.68,  # Character spacing (pixels)
    angle: int = 0,  # Rotation angle (degrees)
    borderstyle: int = 1,  # Border style (1=outline, 3=opaque box)
    outline: int = 0.6,  # Outline thickness (pixels)
    shadow: int = 0,  # Shadow depth (pixels)
    alignment: int = 2,  # Alignment (1-9, 2=center)
    marginl: int = 10,  # Left margin (pixels)
    marginr: int = 10,  # Right margin (pixels)
    # marginv: int = 145,  # Vertical margin (pixels)
    marginv: int = 160,  # Vertical margin (pixels)
):
    subtitle_style = (
        f"Fontname={font},"
        f"Fontsize={fontsize},"
        f"PrimaryColour={primary_colour},"
        f"SecondaryColour={secondary_colour},"
        f"OutlineColour={outline_colour},"
        f"BackColour={back_colour},"
        f"Bold={bold},"
        f"Italic={italic},"
        f"Underline={underline},"
        f"StrikeOut={strikeout},"
        f"ScaleX={scalex},"
        f"ScaleY={scaley},"
        f"Spacing={spacing},"
        f"Angle={angle},"
        f"BorderStyle={borderstyle},"
        f"Outline={outline},"
        f"Shadow={shadow},"
        f"Alignment={alignment},"
        f"MarginL={marginl},"
        f"MarginR={marginr},"
        f"MarginV={marginv}"
    )
    return subtitle_style
