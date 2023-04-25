def postprocess_caption(caption: str, method: str = "all") -> str:
    # Remove any floating special tokens, make sure there's a . at the end, and capitalize the first letter

    caption = caption.replace("<s>", "").replace("</s>", "")
    caption = caption.replace("<pad>", "").replace("</pad>", "")

    # Replace \n with a space
    caption = caption.replace("\n", " ")
    # Replace multiple spaces with a single space
    caption = " ".join(caption.split())

    # Make sure there's a . at the end
    if not caption.endswith("."):
        caption += "."

    # If there are spaces around a period, remove them
    caption = caption.replace(" .", ".")

    # Make sure the first letter in each sentence is capitalized
    if method != "no_caps":
        caption = ". ".join([s[0].upper() + s[1:] for s in caption.split(". ")])

    # Remove any double periods
    caption = caption.replace("..", ".")

    if "truncate" in method:
        # Truncate the caption to the first period
        if "." in caption:
            caption = caption[: caption.index(".") + 1]

    return caption.strip()
