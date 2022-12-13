

def postprocess_caption(caption: str) -> str:
    # Remove any floating special tokens, make sure there's a . at the end, and capitalize the first letter

    caption = caption.replace("<s>", "").replace("</s>", "")
    caption = caption.replace("<pad>", "").replace("</pad>", "")

    # Make sure there's a . at the end
    if not caption.endswith("."):
        caption += "."

    # Make sure the first letter is capitalized
    if not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]

    return caption.strip()
